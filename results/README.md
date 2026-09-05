# Results of record

Each dated directory under `results/` is one end-to-end run of
`scripts/reproduce.sh <date>` on the reference hardware (see `docs/method.md`),
kept in git so that every number quoted in the README and in the paper points
to a file here. Anything else under `results/` (caches, dry runs, new dates)
is ignored by git until it is promoted to a run of record.

| date | what | headline |
|---|---|---|
| `2026-09-05` | eight neutral facts (4 Italian, 4 English), single chained run, real engine check | 7/8 answers reproduced by the quantized engine (p 0.85–0.96); `it_capitale` does not take within 300 steps; replica = F32 engine on 17/17 grafts; zero interference in the merged overlay |

Layout of a run: `report.md` (human summary), `engine_check.json` (every engine
measurement), `summary.json` (replica descents), `merged.pleo` (the overlay of
all facts), `facts/<id>/` (per-fact descents, checkpoints, precondition records),
`routing/` (routing records used by the F32 fidelity check), `window.log`,
`engine_q8.log`, `engine_f32.log`.

The `overlay_path` fields in `summary.json` and `facts/*/*.json` of the
2026-09-05 run were rewritten from absolute to run-relative paths after the run
(the code now writes them relative); no other value was touched.
