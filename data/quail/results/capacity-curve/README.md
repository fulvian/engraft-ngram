# Capacity curve — 24 / 100 / 300 facts

From a separate, earlier experiment series (not the Quail `s0b`/`z0b` cells above): the
same technique applied at three corpus sizes, measuring how accuracy and collateral
damage change as the number of injected facts grows. `capacity_curve.json` holds the
accuracy side; `damage_n24_seed1.json`, `damage_n100.json`, `damage_n300.json` hold the
collateral-damage side.

| facts | top-1, pinned routing | KL damage, mean | damage / quantization floor |
|---|---|---|---|
| 24  | 0.804 | 0.0037 | 1.1x (indistinguishable from the floor within measurement noise; see caveat below) |
| 100 | 0.792 | 0.0059 | 1.8x |
| 300 | 0.821 | 0.0067 | 2.0x |

All three damage numbers remain below the qualitative damage-discussion threshold (0.012)
used elsewhere in this project, at roughly 31%, 49% and 56% of it respectively (not two
orders of magnitude below -- 0.0067/0.012 is about a factor of 2, not 100).

**Important caveat on the n=24 row: mismatched random seeds.** The accuracy value (0.804)
comes from a seed-0 run; the damage value (0.0037) comes from a seed-1 run of a
differently-sized damage sample (`damage_n24_seed1.json`, 27/45 chunks, 22528 positions).
A seed-0 damage measurement also exists on a smaller sample (26 chunks, 20480 positions)
and reads 0.0035 — close to seed 1's 0.0037, i.e. the difference is within the
measurement's own noise, but the two numbers in the table row above are not from the same
run. Do not read the top-1 and damage columns of the n=24 row as two measurements of one
identical trained model instance.

**No quantization-noise floor exists at n=100 or n=300.** The only measured quantization
floor (comparing the IQ4_NL-quantized table against FP8) was measured on the n=24 corpus:
KL 0.0033 (synthetic calibrated noise) and a 0.0079 median weight-delta norm. It is not
valid to compare the n=100/n=300 damage numbers above against that floor without stating
that the floor itself is an n=24 measurement, not a fresh floor at those sizes.

`capacity_curve.json` also carries a `uae_curve` field: top-1 numbers from a separate
comparison baseline at several corpus sizes, kept for reference alongside this technique's
own curve.

## Fields removed from the source data

`capacity_curve.json` rows had `eval_path` and `corpus_path` fields carrying private
relative paths and internal experiment codenames; both removed. The `arm`/`label` fields
(`J`, `h4`, `W1`, `W2`) are internal run identifiers kept as-is; they carry no path or
credential information.
