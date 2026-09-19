# Collateral damage (measure 4) -- summary

Blocks requested: 45, run: 32 (cap 40.0 min).
Total time: 2416.8 s; mean forward: 73.84 s

## Global (random blocks)

- positions: 26624
- dnll: mean=0.0001, median=0.0000
- kl: mean=0.0037, median=0.0005
- fraction kl>0.01/0.1/1: 0.082 / 0.003 / 0.000

## At the hits, by stratum (target rows: `top`/`tail`)

| stratum | n | dnll@hit | kl@hit | dnll +1..+8 | kl +1..+8 | dnll +9..+64 | kl +9..+64 |
|---|---|---|---|---|---|---|---|
| tail | 8 | -0.0425 | 0.0065 | 0.0222 | 0.0083 | 0.0066 | 0.0057 |
| top | 11 | 0.1036 | 0.0094 | -0.0383 | 0.0067 | 0.0062 | 0.0057 |

### Target rows, stratum `top`

| row | reads (/corpus) | dnorm | cos | dnll@hit | kl@hit | kl +1..+8 |
|---|---|---|---|---|---|---|
| 45215359 | 88504 | 0.1570 | 0.4090 | 0.0478 | 0.0028 | 0.0045 |
| 65841408 | 87110 | 0.1568 | 0.3090 | 0.0508 | 0.0049 | 0.0062 |
| 27902507 | 88503 | 0.1542 | 0.4113 | 0.0006 | 0.0037 | 0.0045 |
| 53596120 | 87110 | 0.1540 | 0.3508 | -0.0774 | 0.0202 | 0.0044 |
| 120259304 | 88503 | 0.1471 | 0.3653 | -0.0052 | 0.0002 | 0.0017 |
| 117674008 | 88503 | 0.1463 | 0.4193 | 0.0902 | 0.0149 | 0.0094 |
| 36294323 | 87110 | 0.1483 | 0.3699 | 0.3894 | 0.0240 | 0.0021 |
| 90644285 | 87110 | 0.1481 | 0.2194 | 0.4678 | 0.0223 | 0.0250 |
| 157885351 | 87110 | 0.1470 | 0.2467 | 0.0160 | 0.0039 | 0.0013 |
| 82634375 | 88503 | 0.1428 | 0.4111 | 0.0862 | 0.0024 | 0.0100 |
| 1731449 | 87110 | 0.1450 | 0.4488 | 0.0737 | 0.0047 | 0.0041 |

## Controls (hits on UNCHANGED rows inside a hit block)

- total controls: 160 (upstream of the target row: 40, downstream: 120)
- upstream violations (kl!=0 where 0 is expected, causal locality): 24
- mean kl of downstream controls (expected propagation, no constraint): 0.0063

