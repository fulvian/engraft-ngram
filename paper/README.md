# Technical report

`engraft.tex` is the source of the report; `engraft.pdf` is the committed build.
Every number in the report is taken from files under `../results/2026-09-05/`
(v0.1, see `../results/README.md`) or `../data/quail/results/` (v0.2, the Quail
usage-corpus descent), except four rows of Table 1 (the captured-routing and
released-routing cells without mass weighting, seeds 0 and 1), whose files are
not part of this repository.

Build with [tectonic](https://tectonic-typesetting.github.io/) (downloads the
needed packages on first run, no TeX installation required):

```sh
tectonic engraft.tex
```

or with any TeX Live: `pdflatex engraft.tex` twice. `mechanism.pdf` is the
figure, converted from `../docs/img/mechanism.svg`.

The report is licensed under CC BY 4.0 (`LICENSE` in this directory); the code
in the repository under Apache 2.0.
