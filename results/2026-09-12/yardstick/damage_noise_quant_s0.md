# Collateral damage (measure 4) -- summary

Blocks requested: 45, run: 33 (cap 44.0 min).
Total time: 2643.7 s; mean forward: 78.43 s

## Global (random blocks)

- positions: 26624
- dnll: mean=-0.0003, median=0.0000
- kl: mean=0.0033, median=0.0004
- fraction kl>0.01/0.1/1: 0.076 / 0.003 / 0.000

## At the hits, by stratum (target rows: `top`/`tail`)

| stratum | n | dnll@hit | kl@hit | dnll +1..+8 | kl +1..+8 | dnll +9..+64 | kl +9..+64 |
|---|---|---|---|---|---|---|---|
| tail | 8 | -0.0117 | 0.0044 | -0.0199 | 0.0033 | 0.0011 | 0.0039 |
| top | 12 | -0.0101 | 0.0063 | 0.0141 | 0.0050 | 0.0004 | 0.0060 |

### Target rows, stratum `top`

| row | reads (/corpus) | dnorm | cos | dnll@hit | kl@hit | kl +1..+8 |
|---|---|---|---|---|---|---|
| 45215359 | 88504 | 0.1570 | 0.4090 | -0.0250 | 0.0026 | 0.0092 |
| 65841408 | 87110 | 0.1568 | 0.3090 | 0.0041 | 0.0000 | 0.0015 |
| 27902507 | 88503 | 0.1542 | 0.4113 | -0.0718 | 0.0116 | 0.0035 |
| 53596120 | 87110 | 0.1540 | 0.3508 | -0.0304 | 0.0057 | 0.0036 |
| 120259304 | 88503 | 0.1471 | 0.3653 | 0.0099 | 0.0003 | 0.0025 |
| 117674008 | 88503 | 0.1463 | 0.4193 | -0.2035 | 0.0091 | 0.0055 |
| 36294323 | 87110 | 0.1483 | 0.3699 | -0.1058 | 0.0244 | 0.0009 |
| 90644285 | 87110 | 0.1481 | 0.2194 | 0.1870 | 0.0175 | 0.0242 |
| 157885351 | 87110 | 0.1470 | 0.2467 | 0.0421 | 0.0007 | 0.0012 |
| 82634375 | 88503 | 0.1428 | 0.4111 | 0.0226 | 0.0014 | 0.0056 |
| 1731449 | 87110 | 0.1450 | 0.4488 | 0.0543 | 0.0018 | 0.0010 |
| 150421660 | 88503 | 0.1413 | 0.4038 | -0.0042 | 0.0003 | 0.0015 |

## Controls (hits on UNCHANGED rows inside a hit block)

- total controls: 164 (upstream of the target row: 43, downstream: 121)
- upstream violations (kl!=0 where 0 is expected, causal locality): 27
- mean kl of downstream controls (expected propagation, no constraint): 0.0058

