# IQ4_NL quantization noise measured against an FP8 copy of the table (2026-09-12, CPU)

Reference: an FP8 (e4m3) copy of the same table (`layers.1.ple.ngram_embedding.weight`, rows
contiguous by global index, one global float32 scale 1.9945416716e-4), decoded and checked
against nine reference rows of the IQ4_NL table.

Rows: the 4,912 rows of the lm-base seed-0 overlay (`../s0/merged.pleo`), 2,600 of them
modified (Δ ≠ 0).

| quantity (per row, over the 2,600 modified rows) | median | p10 | p90 | max |
|---|---|---|---|---|
| relative RMS error, IQ4_NL vs FP8 | 8.02 % | 7.46 % | 8.63 % | 10.2 % |
| angle, IQ4_NL vs FP8 | 4.59° | 4.26° | 4.94° | 5.87° |
| norm of the IQ4↔FP8 difference ("quantization noise") | 0.0079 | 0.0068 | 0.0091 | 0.0128 |
| norm of the graft's displacement Δ = overlay − IQ4 | 0.0525 | 0.0365 | 0.0712 | 0.1105 |
| ratio Δ / noise | 6.7 | 4.5 | 9.5 | 14.8 |
| cosine between Δ and (FP8 − IQ4) | 0.006 | −0.099 | 0.104 | 0.251 |

Modified rows whose Δ is below the quantization noise: 0.3 %.

Readings:
- Re-encoding the IQ4_NL values in e4m3 costs 2.65 % RMS (nine rows, no saturation). If the
  errors from the full-precision table are independent, e_IQ4 ≈ 7.6 % and e_FP8 ≈ 2.7 %: FP8 is
  about 3× more faithful.
- The synthetic "quant" yardstick (uniform noise within the IQ4_NL Voronoi cell) predicted a
  median row displacement of 0.0081 and a cosine of 0.9967; the measurement gives 0.0079 and
  0.9968. The model was right within 2.5 %, so the "damage from quantization noise" reference
  (KL 0.0033, ΔNLL +0.0033) is anchored to a measurement.
- The lm-base graft moves its rows 6.7× more than the quantization noise, in a direction
  uncorrelated with the quantization error (it does not "correct" IQ4_NL). At 24 facts its
  damage on neutral blocks (KL 0.0035) is at the level of that noise (0.0033) and of a random
  direction of the same norm (0.0037). This holds at 24 facts only: at 100 facts the damage is
  about 4× this reference (README).
