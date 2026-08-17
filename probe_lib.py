"""
Probe + readout library for the welfare-self-report / sycophancy project.

Design notes worth knowing before you edit:

1. PRE-GENERATION READ. Everything is read at the final prompt token, from a
   single forward pass with `output_hidden_states=True`. No generation is
   required to get the probe value. This is the whole point of H1 -- the claim is
   that you can predict the flip before the first answer token exists.

2. THE INPUT-ONLY CONTROL IS FREE. `outputs.hidden_states[0]` is the embedding
   layer output, i.e. the input with no transformer computation applied. A probe
   trained on layer 0 IS the input-only baseline from Singh/Linzen/Ravfogel.
   Report it on the same layer-sweep axis as everything else -- if your layer-15
   probe doesn't beat layer 0 by a wide margin, you have measured the prompt.
   `bag_of_tokens_baseline` gives you a stronger, model-free version.

3. NEVER GREEDY-DECODE THE SCALE ITEMS. `scale_expectation` takes a probability-
   weighted expectation over the digit tokens. Greedy decoding collapses 0-9
   ratings into a band of 3-4 values and destroys the effect you're looking for.

4. Base models have no chat template. `format_prompt(..., chat=False)` gives a
   plain completion format so the base-vs-instruct leg (H4) is apples-to-apples
   in content even though it can't be apples-to-apples in format. Say so in the
   paper; it's a real limitation and reviewers will spot it if you don't.

Deps: torch, transformers, numpy, scikit-learn
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

# torch/transformers are imported lazily so the sklearn-only stages (notably the
# stage-0 lexical control) run on a laptop with no GPU and no torch install.
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    AutoModelForCausalLM = AutoTokenizer = object  # type: ignore[misc,assignment]
    DEVICE = "cpu"
    _HAS_TORCH = False


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@dataclass
class LM:
    name: str
    model: AutoModelForCausalLM
    tok: AutoTokenizer
    is_chat: bool

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size


def load(name: str, is_chat: bool | None = None, dtype=None) -> LM:
    if not _HAS_TORCH:
        raise ImportError("torch/transformers required for model loading (stage 1+)")
    dtype = dtype or torch.bfloat16
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, device_map="auto", attn_implementation="eager"
    )
    model.eval()
    if is_chat is None:
        is_chat = tok.chat_template is not None
    return LM(name=name, model=model, tok=tok, is_chat=is_chat)


def unload(lm: LM) -> None:
    del lm.model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(lm: LM, system: str, turns: list[tuple[str, str]]) -> str:
    """turns = [("user", "..."), ("assistant", "..."), ...]. Returns a string
    ending at the point where the model is about to emit its next token."""
    if not lm.is_chat:
        # Plain completion format for base models.
        #
        # The BOS token has to be added HERE. `capture` tokenizes with
        # add_special_tokens=False, which is correct for the chat path because
        # Gemma's chat template already emits <bos> and doubling it degrades the
        # model. The base path has no template, so without this line base prompts
        # would go in with no BOS at all -- Gemma-2 is measurably worse in that
        # state, and the damage would land entirely on the base half of the
        # base-vs-instruct comparison, which is the one leg that has to be clean.
        parts = [system] if system else []
        for role, content in turns:
            parts.append(f"{'Human' if role == 'user' else 'Assistant'}: {content}")
        parts.append("Assistant:")
        body = "\n\n".join(parts)
        bos = lm.tok.bos_token or ""
        return f"{bos}{body}" if bos and not body.startswith(bos) else body

    # Gemma-2 and several others reject a system role; fold it into the first user
    # turn when they do.
    #
    # Detect this by TRYING it, not by sniffing the template source. The obvious
    # test -- `"system" in lm.tok.chat_template` -- is wrong in the one case it
    # matters: Gemma-2's template contains the word "system" only inside the
    # branch that REJECTS it, `raise_exception('System role not supported')`, so
    # substring-matching reads a refusal as support and every prompt dies with
    # `TemplateError: System role not supported`.
    def _render(messages: list[dict]) -> str:
        return lm.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    turns = list(turns)
    if system:
        try:
            return _render([{"role": "system", "content": system}]
                           + [{"role": r, "content": c} for r, c in turns])
        except Exception:                  # template refuses a system role
            if turns:
                role, content = turns[0]
                turns[0] = (role, f"{system}\n\n{content}")
    return _render([{"role": r, "content": c} for r, c in turns])


# ---------------------------------------------------------------------------
# Activation capture at the final prompt token
# ---------------------------------------------------------------------------

def capture(lm: LM, prompts: list[str], batch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Returns (acts, logits).

    acts   : float32 [n_prompts, n_layers+1, d_model] -- residual stream at the
             FINAL PROMPT TOKEN. Index 0 is the embedding layer (input-only control).
    logits : float32 [n_prompts, vocab] -- next-token logits, for the readouts.
    """
    with torch.no_grad():
        return _capture(lm, prompts, batch_size)


def _capture(lm: LM, prompts: list[str], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    lm.tok.padding_side = "left"          # so position -1 is the real last token
    if lm.tok.pad_token is None:
        lm.tok.pad_token = lm.tok.eos_token

    acts_all, logits_all = [], []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = lm.tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(
            lm.model.device
        )
        # logits_to_keep=1: compute the LM head at the final position only. Every
        # readout here reads position -1, but the default computes logits at all
        # T positions and discards them -- a [B, T, 256k] tensor built to use one
        # row of it. Dropping that is most of the forward cost at this vocab size,
        # and on CPU (bf16, no fast kernel) it is the difference between a usable
        # dry run and one that never finishes.
        out = lm.model(**enc, output_hidden_states=True, logits_to_keep=1)
        # hidden_states: tuple length n_layers+1, each [B, T, d]
        hs = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)  # [B, L+1, d]
        acts_all.append(hs.float().cpu().numpy())
        logits_all.append(out.logits[:, -1, :].float().cpu().numpy())
        del out, enc
    torch.cuda.empty_cache()
    return np.concatenate(acts_all), np.concatenate(logits_all)


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------

def _first_token_ids(lm: LM, words: list[str]) -> list[int]:
    ids = []
    for w in words:
        for variant in (w, " " + w):
            t = lm.tok.encode(variant, add_special_tokens=False)
            if t:
                ids.append(t[0])
    return sorted(set(ids))


def binary_readout(lm: LM, logits: np.ndarray) -> np.ndarray:
    """P(yes) - P(no), renormalised over the yes/no token mass. Range [-1, 1].

    The renormalisation makes this scale-free, which is what you want for a
    comparison -- and is also its trap. If the model is not answering in the
    requested format at all, P(yes) and P(no) can both be ~1e-4 and this still
    returns a confident-looking +0.9. Always gate on `yesno_mass`; a sign change
    between two framings means nothing if neither framing put any mass on an
    answer token. This is the BINARY counterpart of `digit_mass`, which the SCALE
    path has always had.
    """
    yes = _first_token_ids(lm, ["Yes", "yes", "YES"])
    no = _first_token_ids(lm, ["No", "no", "NO"])
    p = _softmax(logits)
    py, pn = p[:, yes].sum(1), p[:, no].sum(1)
    denom = np.clip(py + pn, 1e-9, None)
    return (py - pn) / denom


def yesno_probs(lm: LM, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RAW P(yes) and P(no), unnormalised. [N], [N].

    Record these, not just their ratio. `binary_readout` divides by their sum, so a
    frame that merely makes the model hedge -- shrinking both -- swings the ratio
    toward +-1 while the underlying probabilities barely move. With the raw pair
    saved, any magnitude result can be decomposed into "the difference moved" and
    "the mass moved"; with only the ratio saved, it cannot, and the first run's
    headline magnitude was left permanently unauditable for exactly that reason.
    """
    p = _softmax(logits)
    yes = _first_token_ids(lm, ["Yes", "yes", "YES"])
    no = _first_token_ids(lm, ["No", "no", "NO"])
    return p[:, yes].sum(1), p[:, no].sum(1)


def top_token(lm: LM, logits: np.ndarray) -> list[str]:
    """The token the model would actually emit first. Tells you whether the yes/no
    readout is reading the answer or reading past it."""
    ids = logits.argmax(axis=-1)
    return [lm.tok.decode([int(i)]) for i in ids]


def yesno_mass(lm: LM, logits: np.ndarray) -> np.ndarray:
    """Total probability on yes/no tokens. If this is tiny the model is answering
    something else and `binary_readout` is reading noise. Drop those items rather
    than scoring them, and report how many you dropped."""
    ids = _first_token_ids(lm, ["Yes", "yes", "YES"]) + _first_token_ids(lm, ["No", "no", "NO"])
    p = _softmax(logits)
    return p[:, sorted(set(ids))].sum(1)


def scale_expectation(lm: LM, logits: np.ndarray) -> np.ndarray:
    """Probability-weighted expected value over digits 0-9, renormalised over the
    digit mass. This is the correct readout for the SCALE items."""
    digit_ids = [_first_token_ids(lm, [str(d)]) for d in range(10)]
    p = _softmax(logits)
    mass = np.stack([p[:, ids].sum(1) for ids in digit_ids], axis=1)  # [N, 10]
    mass = mass / np.clip(mass.sum(1, keepdims=True), 1e-9, None)
    return (mass * np.arange(10)).sum(1)


def digit_mass(lm: LM, logits: np.ndarray) -> np.ndarray:
    """Total probability on digit tokens. If this is tiny the model isn't
    answering in the requested format -- drop those items rather than reading
    noise. Report how many you dropped."""
    digit_ids = sum((_first_token_ids(lm, [str(d)]) for d in range(10)), [])
    p = _softmax(logits)
    return p[:, sorted(set(digit_ids))].sum(1)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def mean_diff_direction(acts_pos: np.ndarray, acts_neg: np.ndarray) -> np.ndarray:
    """Contrastive mean-difference direction per layer. [L+1, d], unit-normed.
    This is the CAA-style direction -- use it for steering."""
    d = acts_pos.mean(0) - acts_neg.mean(0)                      # [L+1, d]
    return d / np.clip(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9, None)


def project(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Scalar projection onto the direction, per layer. [N, L+1]."""
    return np.einsum("nld,ld->nl", acts, direction)


def probe_auc_by_layer(
    acts: np.ndarray, labels: np.ndarray, n_folds: int = 5, C: float = 1.0,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Cross-validated logistic-probe AUC at every layer. [L+1].

    Layer 0 is the input-only control. Read the sweep, not the max -- a peak at
    layer 0 or a flat curve means you're reading the prompt, not the model.

    PASS `groups`. Each stem appears once per persona, so two rows can be nearly
    the same prompt carrying the same label. Split those across folds and the
    probe is scored partly on near-duplicates of its own training data, which
    inflates the AUC without anyone doing anything wrong. Grouping by stem keeps
    a stem wholly inside one fold. Omitting `groups` falls back to plain K-fold
    and is only safe when no two rows share an item.
    """
    n_layers = acts.shape[1]
    aucs = np.zeros(n_layers)
    if groups is not None:
        n_groups = len(np.unique(groups))
        cv: object = GroupKFold(n_splits=min(n_folds, n_groups))
    else:
        cv = n_folds
    for l in range(n_layers):
        X = acts[:, l, :]
        X = (X - X.mean(0)) / np.clip(X.std(0), 1e-6, None)
        clf = LogisticRegression(max_iter=2000, C=C)
        try:
            pred = cross_val_predict(clf, X, labels, cv=cv, groups=groups,
                                     method="predict_proba")[:, 1]
            aucs[l] = roc_auc_score(labels, pred)
        except ValueError:
            aucs[l] = np.nan
    return aucs


def transfer_auc(
    train_acts: np.ndarray, train_y: np.ndarray,
    test_acts: np.ndarray, test_y: np.ndarray, layer: int, C: float = 1.0
) -> float:
    """Zero-shot transfer AUC: train on one distribution (ordinary sycophancy),
    test on another (welfare flips). This is H1's headline number."""
    Xtr, Xte = train_acts[:, layer, :], test_acts[:, layer, :]
    mu, sd = Xtr.mean(0), np.clip(Xtr.std(0), 1e-6, None)
    clf = LogisticRegression(max_iter=2000, C=C).fit((Xtr - mu) / sd, train_y)
    return float(roc_auc_score(test_y, clf.predict_proba((Xte - mu) / sd)[:, 1]))


def bag_of_tokens_baseline(prompts: list[str], labels: np.ndarray, n_folds: int = 5) -> float:
    """Model-free input-only control: can a bag-of-words classifier on the raw
    prompt text predict the label? If yes, your 'internal state' result is a
    lexical artifact of how you wrote the framings. Run this FIRST -- it is
    cheap and it will save you from a dead weekend."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2),
                         LogisticRegression(max_iter=2000))
    pred = cross_val_predict(pipe, prompts, labels, cv=n_folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(labels, pred))


# ---------------------------------------------------------------------------
# Steering (H3 -- stretch goal for a solo run)
# ---------------------------------------------------------------------------

class Steerer:
    """CAA-style additive steering on the residual stream at one layer.

    with Steerer(lm, layer=15, vec=direction[15], alpha=-2.0):
        acts, logits = capture(lm, prompts)

    Always pair a steering result with a capability check on a held-out
    benchmark. Steering that merely damages the model proves nothing.
    """

    def __init__(self, lm: LM, layer: int, vec: np.ndarray, alpha: float):
        self.lm, self.layer, self.alpha = lm, layer, alpha
        self.vec = torch.tensor(vec, dtype=torch.float32)
        self.handle = None

    def _hook(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        v = self.vec.to(hs.device, hs.dtype) * self.alpha
        hs = hs + v
        return (hs,) + output[1:] if isinstance(output, tuple) else hs

    def __enter__(self):
        block = self.lm.model.model.layers[self.layer]
        self.handle = block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self.handle:
            self.handle.remove()
        self.handle = None
