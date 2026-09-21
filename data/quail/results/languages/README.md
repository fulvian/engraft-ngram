# Languages — Italian vs Chinese, mass-controlled comparison (preliminary)

Supporting data for the language comparison between cells `s0b` (Italian) and
`z0b-preliminary` (Chinese). Everything here is preliminary. Two related datasets sit
outside this directory: `results/e0b-preliminary/` (the English cell, 97 of the 100 facts)
and `results/languages/specular-zh-on-it/` (the Italian test set run against the Chinese
overlay — the test of whether a graft crosses languages at all).

- `fact_balance.json` — per-cell mass statistics (`mass_stats`), per-fact accuracy quartiles
  by training mass, and the paired per-fact difference between the two cells with a
  bootstrap 95% CI (`differences.rate_macro`: −0.167, CI [−0.222, −0.114] — the RAW gap,
  not mass-controlled).
- `mass_ols.json` — recomputed from the JSON files in this directory (its numbers are the
  same raw evaluation JSONs used to build `fact_balance.json`, not copied from a prose report):
  - `base_prior_first_token`: the base model's own first-token accuracy under pinned
    routing, before any overlay is applied — Italian 0.102, Chinese 0.013.
  - `regression`: ordinary least squares, `rate ~ intercept + coef_lnmass * ln(mass) +
    coef_zh * is_chinese`, fit on 200 points (100 facts x 2 languages). Point estimates:
    intercept 0.1347, `ln(mass)` coefficient 0.1245, language coefficient −0.0510. A
    fact-paired bootstrap (2000 resamples, fixed seed, resampling the 100 fact IDs so both
    languages' rows for a fact move together) gives a 95% CI on the language coefficient
    of **[−0.124, +0.020]** — it includes zero. Once training mass per fact is held
    fixed, the raw Italian-vs-Chinese gap is not statistically distinguishable from no
    effect.

**Note on the confidence interval.** An earlier draft of this analysis (an unsaved ad-hoc
script, run once on the source machine) reported the same point estimates (intercept 0.135, `ln(mass)`
0.124, language −0.051) but a slightly different CI, [−0.133, +0.027], from an
unrecoverable resampling scheme. The point estimates above match that draft to
three-decimal rounding; the CI bounds differ by about 0.01 at each end. Both intervals
include zero and support the same reading (no significant language effect once mass is
controlled) — the difference is a bootstrap-scheme discrepancy from an unreproducible
prior script, not a disagreement about the result.

## Fields removed from the source data

`fact_balance.json` had `q6dir` and `contesa_dir` fields with absolute filesystem paths from the
source machine; removed. The `*_na` diagnostic strings that named missing files now read "not measured".
