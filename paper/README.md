# Technical report

`engraft.tex` is the source of the report; `engraft.pdf` is the committed build.
Every number in the report is taken from files under `../results/2026-09-05/`
(see `../results/README.md`), except Section 5, which is explicitly labelled as
development observations not yet reproducible from this repository.

Build with [tectonic](https://tectonic-typesetting.github.io/) (downloads the
needed packages on first run, no TeX installation required):

```sh
tectonic engraft.tex
```

or with any TeX Live: `pdflatex engraft.tex` twice. `mechanism.pdf` is the
figure, converted from `../docs/img/mechanism.svg`.

The report is licensed under CC BY 4.0 (`LICENSE` in this directory); the code
in the repository under Apache 2.0.
