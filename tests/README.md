# Tests & benchmarks

## Running the suite

```bash
make test                 # full suite via pytest (uses the repo .venv)
.venv/bin/python -m pytest tests/vesma/test_matcher.py   # a single file
```

GPU/Metal/CUDA and most `perf`/`benchmark` tests skip themselves unless the relevant
backend or environment is available, so a plain `make test` exercises the CPU/RDKit
paths.

### Machines without a working Metal/CUDA backend

Some dev machines report Metal support but it is non-functional (and have no CUDA GPU).
The Metal/CUDA tests then fail for environmental reasons, not regressions. Skip them:

```bash
.venv/bin/python -m pytest tests -k "not metal and not cuda"
```

## Benchmarks

Benchmarks are pytest tests gated behind the `perf` / `paper_benchmark` markers and
driven by the `Makefile` (env vars like `BENCH_TARGET`, `BENCH_MATCH_BACKEND`,
`BENCH_SCALING_SIZES`). Run them through the `bench-*` targets, e.g.:

```bash
make bench-match-rdkit-filtercatalog
make bench-route-replay-rdkit
```

All benchmark targets are pytest-driven. `make bench-folded-fingerprint` runs
`tests/vesma/perf_folded_fingerprint.py::test_bench_folded_fingerprint`; set
`BENCH_FOLDED_FINGERPRINT_CASES="60x200 128x16"` to run multiple sizes.

## Data access

External datasets live under `tests/data/external/<dataset>/{raw,processed}` (+ a
`remix/` tier for datasets combined from several sources). See each dataset's
`README.md` for provenance and licensing.

- Tests reach the tree through the `readonly_datadir` fixture; non-pytest runners use
  `tests.datadirs.external_data_root()`. The tree is read in place (no per-test copy) —
  do not write into it.
- Typed loaders live in `tests/datasets.py`: `load_compatible_templates()` (the
  RDKit/GMol-compatible USPTO rules), `load_retro_benchmark_corpus()`, and the
  PaRoutes n1-d2 x USPTO42k positive-pair loaders. Rule sets are explicit datasets;
  do not create smaller rule sets by slicing `uspto42k-compat` at runtime.
  Parsing/caching is delegated to `src` — these loaders stay thin on purpose.

## Parse cache (joblib, on disk)

`vesma.reaction.reaction_patterns_from_smarts` and `vertsy.stock.load_stock` cache their
results on disk with joblib, under `vesma.loading.cache_dir()`:

- `$XDG_CACHE_HOME/vesma` if `XDG_CACHE_HOME` is set — the `Makefile`'s `BENCH_ENV` sets
  it to `.profile-cache`, so benchmark runs cache under `.profile-cache/vesma`;
- otherwise `./.cache/vesma`.

Templates and stock are parsed once and **reused across invocations** — never
re-parsed until the cache is invalidated. We accept this because the SMARTS parser and
GMol encoding are stable now.

### When and how to invalidate

Invalidate whenever the parser or the GMol/stock encoding changes — otherwise you will
read stale parsed objects:

- **Principled (preferred):** bump the cache version in `src` — `REACTION_CACHE_VERSION`
  in `src/vesma/reaction.py` and/or `STOCK_CACHE_VERSION` in `src/vertsy/stock.py`. The
  version is part of the cache key, so old entries are bypassed automatically.
- **Blunt:** delete the cache directory for a guaranteed clean run, e.g.
  `rm -rf .profile-cache/vesma` (or `rm -rf .cache/vesma`). If a test hits a tricky,
  hard-to-explain mismatch, clearing the cache is the first thing to try.
