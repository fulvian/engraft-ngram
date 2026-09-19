# Corpus sferoglifica -- it

- facts: 24 (shared: 6, unique: 18)
- documents: 3
- facts per class: place=8, instrument=8, scholar=8
- tokens per document: it_0=404, it_1=408, it_2=409
- tokens per answer (identifiers kept as produced, `luogo`=place, `strumento`=instrument,
  `studioso`=scholar): it_luogo_00=2, it_strumento_01=4, it_studioso_02=2, it_luogo_03=4, it_strumento_04=2, it_studioso_05=4, it_luogo_06=4, it_strumento_07=2, it_studioso_08=3, it_luogo_09=2, it_strumento_10=3, it_studioso_11=4, it_luogo_12=2, it_strumento_13=3, it_studioso_14=3, it_luogo_15=3, it_strumento_16=3, it_studioso_17=2, it_luogo_18=3, it_strumento_19=4, it_studioso_20=3, it_luogo_21=2, it_strumento_22=2, it_studioso_23=3

## Single document `it_all` (2026-09-11)

`docs/it_all.txt` / `doc_tokens_it_all.json` = common introduction (88 tokens, once)
+ the three fact blocks of `it_0` (places), `it_1` (instruments), `it_2` (scholars):
434 tokens, built by concatenating the token lists (no re-tokenization). This is "the
document" of the teacher for stage 0: the usage corpus covers all three relations, and
with a single partial document the teacher would coincide with the base on two thirds
of the fragments.

## Glossary of the Italian keys in `census_it.json`

`census_it.json` is kept byte for byte as written by the run of record; some of its keys are Italian.

| key | meaning |
|---|---|
| `affermazione`, `domanda`, `parafrasi` | families: statement, question, paraphrase |
| `altro` | other |
| `cavallo` | straddling (a window that spans two segments) |
| `entita` | entity |
| `inizio`, `termine` | start, end (of the subject or term span) |
| `n_fatti`, `n_righe`, `n_letture` | number of facts, of rows, of reads |
| `righe_distinte` | distinct rows |
| `letture_per_riga_media` | mean reads per row |
| `istogramma_letture_per_riga` | histogram of reads per row |
| `n_congelate`, `frazione_congelata` | number and fraction of frozen rows |
| `n_entita_shared_B`, `n_entita_shared_B_congelate` | entities shared in column B, and how many of them are frozen |
| `frazione_entita_shared_B_sopravvive` | fraction of shared column-B entities that survive the freeze |
| `share_con_c0`, `share_senza_c0` | share with and without the C0 contexts |
| `nota` | note |

In fact ids, `luogo`, `strumento` and `studioso` mean place, instrument and scholar.
