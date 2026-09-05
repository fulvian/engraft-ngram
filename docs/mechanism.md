# The n-gram (PLE) table: addressing and reading

A Qwen4Exp-family model has one special layer (`ple_layer` in its
hyperparameters, layer 1 in the models this repository targets) that, for
every position, looks up 16 rows in a large lookup table instead of (or in
addition to) computing from the residual stream. `engraft/table.py` reads
that table offline, directly from its GGUF, and reproduces the same
addressing the engine uses internally.

## Layout on disk

The table GGUF stores one tensor per head, `ple_ngram_embd.{h}.weight`, type
IQ4_NL, shape `[160, p_h]` (`p_h` the head's local vocabulary size). Rows of
head `h` are contiguous in the file. `PleTable.read_rows_raw` seeks to the
right offset and reads exactly the requested rows -- never the whole tensor.

IQ4_NL quantizes 32 values into an 18-byte block: a float16 scale `d`
followed by 16 packed nibbles, each indexing a 16-entry codebook (values
roughly in [-127, 113], never including 0). `dequant_iq4nl` turns raw bytes
into float32 rows; `row_is_zero` exploits the fact that the codebook has no
zero entry to detect an all-zero row from its scale alone, without
dequantizing.

## Which 16 rows: the address hash

At each position `t`, the context is the current token and the previous
`ngram_size - 1` tokens (`ngram_size = 3` for a trigram table): `ctx = [x_t,
x_{t-1}, x_{t-2}]`. A missing predecessor (start of sequence) or an EOS found
while walking backward zeroes it and every older position in the context,
substituting the model's EOS token id -- but the EOS of the *current* token
does not cut its own context.

For each n (2, then `ngram_size`), the mixed hash is:

```
mixed = ctx[0] * m[0] ^ ctx[1] * m[1] ^ ... ^ ctx[n-1] * m[n-1]   (mod 2^64)
```

using the model's own `layer_multipliers`. `heads_per_ngram` heads per n
(8 heads for the bigram hash, 8 more for the trigram hash in the models this
repository targets) each take `mixed % head_vocab_sizes[h]` as their local
row index. `PleTable.ngram_addresses` reproduces this for a whole token
sequence at once; `engraft.lens.RowSet.from_position` wraps it together with
the dequantized rows and their *global* indices (local index plus the head's
offset in the merged row space, `local_to_global`).

## Overlays: never touching the quantized bytes

An overlay never rewrites the GGUF. Instead, `.pleo` files (written and read
by `engraft.lens`) list a set of global row indices together with float32
replacement vectors; the engine fork's gather path substitutes them in at
read time (see `engine/README.md`). A graft (`engraft.replica.graft`) is,
mechanically, the process of finding replacement vectors for a fact's own 16
rows at each answer position and writing them out as one `.pleo` overlay.
