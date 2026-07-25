# Data card

## Source

- **Real runs:** [WikiText-103](https://huggingface.co/datasets/wikitext)
  (`wikitext-103-raw-v1`), ~103M tokens of verified Wikipedia articles.
  License: **CC BY-SA 3.0**. Loaded via `datasets` streaming (install `[data]`).
- **Offline debug/tests:** `examples/sample_corpus.txt`, a tiny hand-written corpus
  tokenized with the dependency-free byte tokenizer.

## Preprocessing

1. **Tokenize** with GPT-2 BPE plus an appended `[MASK]` absorbing token
   (`vocab_size = 50258`, `mask_token_id = 50257`, `eos_token_id = 50256`).
2. **Pack**: concatenate documents into a single token stream, one EOS token
   between documents, then chunk into fixed-length blocks of `block_size` (1024).
   The trailing partial block is dropped.
3. **Split**: the first `val_blocks` (256) blocks are held out for validation; the
   rest are training. An absolute count keeps the split deterministic while
   streaming, without buffering the whole corpus.
4. **Store**: `.npy` shards of shape `(num_blocks, block_size)` in the smallest
   unsigned dtype that fits the vocab (`uint16` for GPT-2), memory-mapped at load.

Masking is **not** baked into the data; the diffusion loss renoises clean blocks
freshly each step, so a block is seen at many noise levels across training.

## Versioning

Each shard is content-addressed: its filename embeds the SHA-256 of its raw bytes,
and `manifest.json` records every shard hash plus a single `dataset_hash`
fingerprint over the tokenizer, block size, val split, and shard hashes (not the
source path, so it is location-independent). Two
identical builds produce an identical `dataset_hash` — the value a training run
cites for reproducibility.

## Manifest fields

`tokenizer`, `tokenizer_version`, `vocab_size`, `mask_token_id`, `eos_token_id`,
`block_size`, `storage_dtype`, `source`, `split`, `val_blocks`, `seed`,
`num_train_blocks`, `num_val_blocks`, `shards[]` (`filename`, `sha256`,
`num_blocks`, `split`), `dataset_hash`.
