# Paper benchmark data

The current complete reaction-application benchmark lives in `reaction-application/`.
Each pytest-benchmark JSON contains one backend, host, workload size, and three timing samples after one warmup. Hardware-labelled files keep the RTX 4090 and RTX 5090 runs separate.

The paper reads these records directly. It requires unique backend/host/size records, matching input hashes, and matching VeSMA processing and precursor counts. To repeat a cell, remove its existing JSON first; duplicate cells fail rendering rather than silently selecting a run. The main table selects the retained 4090 and 5090 records by device metadata rather than a fixed host CPU name. Keep only the intended current host for each GPU.

```sh
make bench-application-rdkit  # serial and parallel
make bench-application-cpu   # serial through 100; parallel through 2881
make bench-application-metal
make bench-application-cuda
```

Defaults: `APPLICATION_SIZES="1 10 100 1000 2881"`, `BENCH_ROUNDS=3`, `BENCH_WARMUP_ROUNDS=1`. Results default to `docs/paper/benchmarks/reaction-application`. Override `APPLICATION_DIR` only for exploratory runs; publication timings belong in the default paper directory, not `docs/reports` or `output`. Runs on a remote pod write there on that pod and must be synced back before a local paper render. Run backends sequentially on an idle machine.

The timed contract `streamed-unkeyed-host-v1` includes matching, graph editing, splitting, full perception, and usable host output. Each batch is consumed and released. Parsing, compilation, keys, canonical SMILES, and deduplication are excluded. RDKit uses native product sanitization. Raw diagnostics and source/input provenance remain in each JSON.

`bench-cpu`, `bench-rdkit`, `bench-metal`, and `bench-cuda` are convenience suites combining their match-only and current application targets (and CUDA route replay). There is no separate split timing target. Obsolete match-plus-split and pre-fix timing records have been removed.

Other directories contain distinct validation or match-only workloads. Match-only is not complete reaction application. Route replay, route recovery, molecular-key audits, and template compatibility are correctness/integration evidence, with their own Make targets.
