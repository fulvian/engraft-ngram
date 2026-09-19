# Collateral damage (measure 4) -- summary

Blocks requested: 45, run: 27 (cap 33.0 min).
Total time: 2049.8 s; mean forward: 74.21 s

Routing of the student forward: **free** (default, live routing).

## Global (random blocks)

- positions: 22528
- dnll: mean=0.0004, median=0.0000
- kl: mean=0.0037, median=0.0003
- fraction kl>0.01/0.1/1: 0.082 / 0.003 / 0.000

## At the hits, by stratum (target rows: `top`/`tail`)

| stratum | n | dnll@hit | kl@hit | dnll +1..+8 | kl +1..+8 | dnll +9..+64 | kl +9..+64 |
|---|---|---|---|---|---|---|---|
| tail | 8 | -0.0085 | 0.0118 | 0.0019 | 0.0076 | -0.0024 | 0.0049 |
| top | 8 | 0.0289 | 0.0106 | -0.0149 | 0.0067 | -0.0009 | 0.0074 |

### Target rows, stratum `top`

| row | reads (/corpus) | dnorm | cos | dnll@hit | kl@hit | kl +1..+8 |
|---|---|---|---|---|---|---|
| 45215359 | 88504 | 0.1570 | 0.4090 | 0.0197 | 0.0058 | 0.0046 |
| 65841408 | 87110 | 0.1568 | 0.3090 | 0.0158 | 0.0045 | 0.0033 |
| 27902507 | 88503 | 0.1542 | 0.4113 | 0.0142 | 0.0029 | 0.0026 |
| 53596120 | 87110 | 0.1540 | 0.3508 | 0.0241 | 0.0076 | 0.0025 |
| 120259304 | 88503 | 0.1471 | 0.3653 | 0.0024 | 0.0003 | 0.0034 |
| 117674008 | 88503 | 0.1463 | 0.4193 | 0.0378 | 0.0270 | 0.0087 |
| 36294323 | 87110 | 0.1483 | 0.3699 | -0.1972 | 0.0162 | 0.0010 |
| 90644285 | 87110 | 0.1481 | 0.2194 | 0.3140 | 0.0204 | 0.0271 |

## Controls (hits on UNCHANGED rows inside a hit block)

- total controls: 138 (upstream of the target row: 36, downstream: 102)
- upstream violations (kl!=0 where 0 is expected, causal locality): 21
- mean kl of downstream controls (expected propagation, no constraint): 0.0087

