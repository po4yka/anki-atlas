# Fuzzing

This repository keeps fuzzing isolated in the root [`fuzz/`](fuzz) package so normal workspace commands stay unchanged.

## Install

```bash
rustup toolchain install nightly
cargo install cargo-fuzz
```

## Run

```bash
cargo +nightly fuzz run anki_reader_normalizer
cargo +nightly fuzz run card_slug
cargo +nightly fuzz run obsidian_frontmatter
cargo +nightly fuzz run generator_apf_linter
cargo +nightly fuzz run jobs_redis_url
cargo +nightly fuzz run indexer_sparse_vector
cargo +nightly fuzz run validation_quality
```

For a quick smoke test:

```bash
cargo +nightly fuzz run anki_reader_normalizer -- -max_total_time=60
```

Tracked seed corpora live under [`fuzz/corpus/`](fuzz/corpus). Generated artifacts and build output stay untracked.

## Reproduce A Crash

When libFuzzer finds a crash it writes an artifact path under `fuzz/artifacts/<target>/...`.

Re-run the target against that artifact:

```bash
cargo +nightly fuzz run anki_reader_normalizer fuzz/artifacts/anki_reader_normalizer/crash-...
```

## Promote A Fix

1. Minimize the crashing input with `cargo +nightly fuzz tmin <target> <artifact>`.
2. Add the minimized reproducer to the matching `fuzz/corpus/<target>/` directory.
3. Add a deterministic regression test in the owning crate before merging the fix.
