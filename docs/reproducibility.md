# Reproducibility

## Rebuild the dataset

Offline (no downloads, uses the shipped sample corpus):

```bash
make data-debug            # aether-prepare data=local_debug
```

Real WikiText-103 (needs the data extra and network access):

```bash
pip install -e ".[data]"
make data                  # python -m aether.data.prepare data=wikitext103
```

Override any field on the command line (Hydra):

```bash
python -m aether.data.prepare data=wikitext103 data.block_size=512 data.max_documents=1000
```

## Verify a build is reproducible

`prepare` prints a `dataset_hash`. Run it twice and confirm the hash is identical;
it is computed from the tokenizer, block size, val split, and the SHA-256 of
every shard's bytes, so any change to the data changes the hash.

```bash
python -m aether.data.prepare data=local_debug | grep dataset_hash
python -m aether.data.prepare data=local_debug | grep dataset_hash   # same value
```

## What determinism relies on

- Fixed source document order + fixed tokenizer + fixed `block_size` -> identical
  packed blocks -> identical shard bytes -> identical hashes.
- Data-loading shuffle is seeded by `seed + epoch` in `DiffusionDataModule`, so the
  first batch of any epoch is byte-for-byte reproducible.
