#!/usr/bin/env bash
# Remote bootstrap for the welfare_probe Lambda box. Idempotent (marker .setup_ready), then execs
# whatever job command it is handed.
#
# WHY THIS DOES NOT BUILD A VENV WITH A PINNED TORCH. The sibling project's runner installs torch
# from the cu124 index into a plain venv, because on its 2026-06 image the default PyPI torch
# resolved too new for the driver and the run silently fell back to CPU. That pin is now the
# failure rather than the fix: measured on this image (2026-08-14, gpu_1x_a100_sxm4),
#
#     Python 3.10.12 | driver 570.148.08 | CUDA 12.8 | preinstalled torch 2.7.0+cu128, cuda True
#
# there is no cu124 wheel set that resolves, and the install dies with ResolutionImpossible before
# a single model is pulled. The image already ships a working CUDA torch, so the right move is to
# USE it and add only the missing pure-Python packages on top.
#
# transformers and accelerate are installed WITH their dependencies. Neither declares a dependency
# on torch, so the resolver cannot replace the working CUDA build with a CPU wheel -- which is the
# only way the documented "silent CPU fallback" trap could bite here. Installing --no-deps would
# trade that non-risk for a real one (a missing transitive dep failing mid-run, after the model
# download). The assertion below is what actually guards the outcome either way.
#
# The CUDA assertion is the load-bearing part and is re-run on EVERY call, warm or cold: a silent
# CPU fallback would bill GPU hours to produce CPU numbers that look identical in the artifact.
#
#   usage:  HF_TOKEN=hf_... bash remote_run.sh bash onbox_h1.sh
set -euo pipefail
cd ~/welfare_probe

if [ ! -f .setup_ready ]; then
  echo "[setup] using preinstalled torch; adding transformers/accelerate/scikit-learn"
  python3 -m pip install -q --upgrade pip || true
  # --upgrade is load-bearing, and so is naming pillow and numpy explicitly. The image ships
  # numpy 1.21.5, scikit-learn 0.23.2 and a pre-9.1 Pillow in /usr/lib/python3/dist-packages;
  # they satisfy transformers' loose pins, so a plain install leaves them in place and the run
  # dies deep in an import chain -- measured, 2026-08-14:
  #   AttributeError: module 'PIL.Image' has no attribute 'Resampling'
  #     -> ModuleNotFoundError: Could not import module 'Gemma2ForCausalLM'
  # transformers reaches PIL through its object-detection loss module, so an image dependency
  # breaks a text-only run. --user installs land in ~/.local, which precedes dist-packages on
  # sys.path, so upgrading here shadows the stale system copies without touching the image.
  # numpy is PINNED to the last 1.x, not upgraded. The preinstalled torch 2.7.0 is built against
  # the numpy 1.x C ABI, so pulling numpy 2.x breaks it -- measured, 2026-08-14:
  #   AttributeError: _ARRAY_API not found / ImportError: numpy.core.multiarray failed to import
  # This is the numpy ABI trap the sibling project documented, reached from the other direction:
  # there by installing numpy too late, here by installing it too new. 1.26.4 is new enough for
  # current scikit-learn and transformers and old enough to keep torch's ABI intact.
  python3 -m pip install -q --upgrade pillow "jinja2>=3.1" "numpy==1.26.4" scikit-learn transformers accelerate
  touch .setup_ready
fi

python3 - <<'PY'
import sys
try:
    import torch
except Exception as e:                       # noqa: BLE001
    print("[setup] FATAL: torch unimportable:", e, file=sys.stderr); sys.exit(3)
if not torch.cuda.is_available():
    print("[setup] FATAL: torch.cuda.is_available() is False -> the run would SILENTLY use CPU.",
          "torch:", torch.__version__, file=sys.stderr)
    sys.exit(3)                              # loud, non-zero: RUN_DONE records it, no results produced
print("[setup] torch", torch.__version__, "cuda", torch.version.cuda,
      "device", torch.cuda.get_device_name(0))
import numpy, sklearn, transformers, PIL, jinja2   # import-ability check (fatal if an ABI clash bit)
# jinja2 >= 3.1 is required by apply_chat_template, and ONLY the chat path touches it -- so a stale
# jinja2 lets a base run finish clean while the instruct run of the same pair dies on prompt 1.
# Measured 2026-08-14: image ships 3.0.3 -> "ImportError: apply_chat_template requires jinja2>=3.1.0".
_jv = tuple(int(x) for x in jinja2.__version__.split(".")[:2])
if _jv < (3, 1):
    print("[setup] FATAL: jinja2", jinja2.__version__, "< 3.1 -> chat templates unusable", file=sys.stderr)
    sys.exit(3)
print("[setup] transformers", transformers.__version__, "numpy", numpy.__version__,
      "scikit-learn", sklearn.__version__, "pillow", PIL.__version__)

# Resolve the CONCRETE model class, not just the package. transformers imports lazily, so
# `import transformers` succeeds even when the Gemma-2 module cannot load; the failure then
# surfaces inside from_pretrained, i.e. after the box has been paid for and the checkpoint
# pulled. Forcing the same lookup here turns that into a five-second setup failure.
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES  # noqa: F401
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM  # noqa: F401
print("[setup] Gemma2ForCausalLM resolvable")
PY

# gated google/gemma-2-9b{,-it} need the HF token; the launcher passes it in the run env.
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out results
echo "[run] $*"
exec "$@"
