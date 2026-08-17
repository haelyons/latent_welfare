# docs

Write-ups for [latent_welfare](../../tree/main). This branch adds `docs/` to the
`main` tree; the experiments and results are unchanged from `main`.

| file | what it is |
|---|---|
| `submission/XPLORE_FROM_LATENTS_intro_f_ii.md` | the submitted report, in markdown |
| `submission/XPLORE_FROM_LATENTS_submission.docx` | the same report on the sprint template |
| `submission/XPLORE_FROM_LATENTS_submission.pdf` | render of the docx |
| `submission/build_docx.py` | populates the template from the markdown; needs `python-docx` |
| `submission/drafts/` | earlier versions, kept because the argument changed shape several times |
| `REPORT.md` | claim-by-claim technical record, every scoped caveat in place |
| `SUMMARY.md` | stage-by-stage narrative, written cold from the records |
| `POST.md` | long-form version of the argument |
| `report_intro.md` | an earlier standalone framing of the question |
| `planning-README.md` | the pre-sprint plan, kept for the record; its hypothesis did not survive |

`REPORT.md`, `SUMMARY.md` and `POST.md` are the companion documents the report
refers to. Numerals in them trace to saved artifacts on `main`:

```bash
python3 index.py --trace docs/REPORT.md --root results
```
