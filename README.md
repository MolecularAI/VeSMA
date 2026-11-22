# VeSMA

VeSMA is a Python library for applying reaction SMARTS and SMIRKS templates to batches of molecules. It enumerates subgraph assignments, applies graph edits, constructs precursor molecules, and records which product, rule, and application produced each precursor.

VeSMA provides NVIDIA CUDA and Apple Metal implementations together with a readable CPU reference, all behind a common interface. Optional fingerprint prefiltering, molecular keys, deduplication, stock lookup, and retrosynthesis search are separate, composable operations.

## Installation

The project requires Python 3.14 or later. Development uses a local environment:

```bash
uv sync --extra dev
```

Common installations are:

```bash
# CPU core
python -m pip install vesma

# CPU plus RDKit interoperability/reference tools
python -m pip install 'vesma[rdkit]'

# NVIDIA CUDA 12 through CuPy (Linux x86-64)
python -m pip install 'vesma[cuda]'
```

For CUDA, install a compatible NVIDIA driver before the `cuda` extra. Metal is available on supported Apple macOS systems without a Python extra; the backend compiles its Objective-C++/Metal bridge on first use with Apple's `clang++` and frameworks.

For CUDA development from this repository, run `make sync-cuda`. It selects Python 3.14,
which has wheels for the locked CuPy release; a fresh uv environment otherwise may select
a newer Python that CuPy does not yet support.

## Quick start: reaction-template application

```python
from vesma import (
    cmol_from_smiles,
    reaction_patterns_from_smarts,
    smiles_from_cmol,
)
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.matcher import ReactionMatcher

reaction_smarts = [
    # Disconnect the bond between an ester oxygen and its alkyl group.
    "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])-[O:3].[C:4]"
]
rules = reaction_patterns_from_smarts(reaction_smarts)
matcher: ReactionMatcher = CpuReactionMatcher(rules)

product_smiles = [
    "CC(=O)OC",  # Methyl acetate: one ester bond to disconnect.
    "COC(=O)CC(=O)OC",  # Dimethyl succinate: two possible disconnections.
]
products = [cmol_from_smiles(smiles)[0] for smiles in product_smiles]

precursors, hits = matcher.match(products)

for product_id, rule_id, application_id, precursor_id in hits:
    print(
        f"product={product_smiles[product_id]}, rule={rule_id}, "
        f"application={application_id}, "
        f"precursor={smiles_from_cmol(precursors[precursor_id])}"
    )
```

`vesma.matcher.ReactionMatcher` is the common protocol. All implementations expose `match(mols, pairs=None) -> (precursors, hits)`. Omitting `pairs`, as above, evaluates the molecule--rule Cartesian product. Passing explicit `(molecule_index, rule_index)` pairs evaluates only those pairs.

Each row of the `(N, 4)` integer `HitArray` is one precursor of one reaction application: its input product index, rule index, application index, and an index into the returned precursor batch. An application is one matched location at which a rule was applied within a product--rule pair. One application can produce multiple co-precursors, so those precursors share the first three columns. Above, methyl acetate has application `0`, while the two ester sites in dimethyl succinate are applications `0` and `1`.

Because the precursor index is explicit, the precursor batch is independent of the hit count: a backend may return one precursor per hit, or deduplicate identical precursors within the batch and have several hit rows point at one batch row. Near-synonymous rules produce identical precursors, so deduplicating can shrink the batch substantially. Consumers therefore index the batch (`precursors[precursor_id]`) rather than zipping it against the hits. The backends above all return one precursor per hit today. The matcher returns raw applications without ranking or performing stock lookup.

The main VeSMA constructors are:

| Backend | Constructor | Molecule/result representation |
|---|---|---|
| CPU reference | `vesma.cpu_matcher.CpuReactionMatcher(rules)` | `CMol` input; raw `CMol` precursor array |
| CUDA | `vesma.cuda_matcher.CudaReactionMatcher(rules, frontier_mode=...)` with `"device"` or `"host"` | `CMol` or packed CUDA input; packed CUDA or host precursor batch |
| Metal | `vesma.metal_reaction_matcher.MetalReactionMatcher(rules)` | `CMol` or packed Metal input; packed host precursor batch |

`CudaFoldedPrefilteredReactionMatcher` and `MetalFoldedPrefilteredReactionMatcher` wrap each matcher in a prefilter. The prefilter can remove impossible pairs but does not choose or rank promising reactions.

## VeSMA-Retrosynthesis demo

This section describes how to run retrosynthesis search demo and make a route report.

Install [Quarto](https://quarto.org) to compile PDF from report's Quarto Markdown (*.qmd).

Ensure RDKit and paper dependencies are installed to render the molecules in the report:

```bash
uv sync --extra dev --extra rdkit --extra paper
```

Example needs stock. Fetch, for example, Enamine Building Blocks from [ChemDOTS deposition](https://zenodo.org/records/10776787):

```bash
make tests/data/external/chemdots-stock/processed/chemdots_enamine_bbs_annotated.smi.bz2
```

The first run have to compute molecular deduplication keys and similarity fingerprints for the full stock, and compile the templates. Keys, fingerprints, and compiled templates are cached, subsequent runs reuse computed values. Expect about 10 minutes of preprocessing on the first run, subsequent cache reading takes just a few seconds.

```bash
# Search for the first 10 targets in file pistachio_hard-targets.smi
# Output goes to output/benchmarks/aostar/host/<timestamp>/report.pdf.
make bench-route-pipeline \
  AOSTAR_HOST_TARGET_LIMIT=10 \
  AOSTAR_HOST_TARGET_OFFSET=0 \
  AOSTAR_HOST_TARGETS_FILE=tests/data/external/paroutes/processed/n1-targets.smi.gz \
  AOSTAR_HOST_RULE_SET=medchem,protecting-groups,hartenfeller \
  AOSTAR_HOST_STOCK=enamine-annotated,common-reagents \
  AOSTAR_HOST_EFFORTS=16x8 \
  AOSTAR_HOST_BACKENDS=metal \
  AOSTAR_HOST_VETOES=withhold \
  AOSTAR_HOST_PENALTY_SITES=steps \
  AOSTAR_HOST_PENALTY_COMPETITORS=none \
  AOSTAR_HOST_HEURISTIC=atoms \
  AOSTAR_HOST_ROUTE_CAP=3 \
  BENCH_ROUNDS=1 \
  BENCH_WARMUP_ROUNDS=0
```


## Tests and benchmark reproduction

```bash
# Full CI-compatible check: lint, formatting, types, and tests
make check

# Fast default tests only. Also `make test`
python -m pytest tests/

# Publication-snapshot CPU validation
make bench-paper-validation

# Backend benchmark families (run only on the corresponding hardware)
make bench-rdkit
make bench-cpu
make bench-cpu-match  # CPU match-only, without reaction application
make bench-application-cpu  # Streamed, unkeyed host output
make bench-application-rdkit  # Serial and parallel RDKit
make bench-application-cuda
make bench-application-metal

# Constrained topology replay: applies the reference rule at each reaction node
make bench-route-replay-rdkit
make bench-route-replay-cuda
make bench-route-replay-metal

# Full-library, depth-limited route-recovery stress test
make bench-route-recovery-used14k-full

# Separate limited-target fanout-depth benchmark
make bench-fanout-depth-cuda
```

All publication results use `pytest-benchmark` JSON, with paper-related records in `benchmark.extra_info`. Retained raw JSON lives under `docs/paper/benchmarks/`. The paper and supporting information render with:

```bash
make paper-render
make paper-si-render
```

## Citation

If you use VeSMA, please cite the accompanying paper:

Alexey Voronov and Anna Reymer. *VeSMA: GPU-native library for retrosynthetic expansion and beyond*. ChemRxiv, 2026.

- Preprint: `CHEMRXIV_URL_PENDING_RELEASE`
- Preprint DOI: `CHEMRXIV_DOI_PENDING_RELEASE`
- Zenodo code deposition DOI: [10.5281/zenodo.24397958](https://doi.org/10.5281/zenodo.24397958)

## License

VeSMA is released under the [Apache-2.0 License](LICENSE).
