# Descent inputs for s0b — not included in this release

Three files that feed the s0b descent and its evaluation are not copied into this repository:

| file | size | used by |
|---|---|---|
| `target.npz` | 262 MB | the descent (`--targets`) |
| `routing.npz` | 87 MB | the descent (`--routing-base`) |
| `test_routing.npz` | 19 MB | the replica evaluation of the test split |
| **total** | **~368 MB** | |

They are left out because they are large. They can be regenerated from the corpus published in
`data/quail/corpus/s0b/` and the model.

## Regenerating

The s0b cell uses the `lm-base` arm (`data/quail/config/s0b.json`): the targets are those of the
**base model**, with no document in the prefix. One pass of `engraft.teacher` over the usage corpus
writes the targets, the base routing and the bias report together; a second pass over the test
corpus writes the test routing:

```
python -m engraft.teacher --k 256 --seed 0 --dense-dtype bf16 \
  --usage-corpus data/quail/corpus/s0b/usage_corpus_resolved.json \
  --out target.npz --routing-out routing.npz --bias-out bias.json

python -m engraft.teacher --k 256 --seed 0 --dense-dtype bf16 \
  --usage-corpus data/quail/corpus/s0b/test_corpus_resolved.json \
  --out test_target.npz --routing-out test_routing.npz --bias-out test_bias.json
```

The measured run captured the routing with the private fast expert kernel. The public reference
kernel gives bit-identical expert logits, so the captured routing is expected to match; a
regenerated `routing.npz` has not yet been compared with the original byte for byte.
