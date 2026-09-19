# Collateral damage (measure 4) -- summary

Blocks requested: 45, run: 26 (cap 33.0 min).
Total time: 2001.9 s; mean forward: 75.33 s

Routing of the student forward: **free** (default, live routing).

## Global (random blocks)

- positions: 20480
- dnll: mean=0.0003, median=0.0000
- kl: mean=0.0035, median=0.0003
- fraction kl>0.01/0.1/1: 0.077 / 0.003 / 0.000

## At the hits, by stratum (target rows: `top`/`tail`)

| stratum | n | dnll@hit | kl@hit | dnll +1..+8 | kl +1..+8 | dnll +9..+64 | kl +9..+64 |
|---|---|---|---|---|---|---|---|
| tail | 8 | -0.0566 | 0.0088 | 0.0115 | 0.0105 | -0.0007 | 0.0047 |
| top | 8 | -0.0143 | 0.0110 | -0.0093 | 0.0043 | 0.0042 | 0.0072 |

### Target rows, stratum `top`

| row | reads (/corpus) | dnorm | cos | dnll@hit | kl@hit | kl +1..+8 |
|---|---|---|---|---|---|---|
| 45215359 | 88504 | 0.1570 | 0.4090 | -0.0186 | 0.0011 | 0.0033 |
| 65841408 | 87110 | 0.1568 | 0.3090 | 0.0109 | 0.0045 | 0.0021 |
| 27902507 | 88503 | 0.1542 | 0.4113 | -0.0371 | 0.0084 | 0.0020 |
| 53596120 | 87110 | 0.1540 | 0.3508 | -0.0792 | 0.0206 | 0.0034 |
| 120259304 | 88503 | 0.1471 | 0.3653 | 0.0038 | 0.0004 | 0.0039 |
| 117674008 | 88503 | 0.1463 | 0.4193 | -0.0107 | 0.0085 | 0.0065 |
| 36294323 | 87110 | 0.1483 | 0.3699 | -0.2994 | 0.0237 | 0.0023 |
| 90644285 | 87110 | 0.1481 | 0.2194 | 0.3157 | 0.0208 | 0.0108 |

## Controls (hits on UNCHANGED rows inside a hit block)

- total controls: 138 (upstream of the target row: 36, downstream: 102)
- upstream violations (kl!=0 where 0 is expected, causal locality): 21
- mean kl of downstream controls (expected propagation, no constraint): 0.0078

