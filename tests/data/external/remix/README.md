Remix datasets.

This tier holds datasets that are *combined or derived from multiple* other
external datasets in this tree, rather than downloaded from a single upstream
source. Their committed artifacts are reproducible from the source datasets via the
generator script that lives next to each one.

Datasets:
- `retro-benchmark/`: a pinned CUDA-matcher benchmark built from the ChEMBL 50k
  subset (`../chembl/`) and the PaRoutes USPTO unique templates (`../paroutes/`).
- `uspto42k-compat/`: the PaRoutes USPTO unique templates filtered to full
  reactions that parse with both RDKit and Vesma and whose LHS fits the GPU atom
  budget.
- `paroutes-n1-d2-match-uspto42k/`: RDKit FilterCatalog positive match pairs for
  PaRoutes `n1-d2-f0` molecules against the full `uspto42k-compat` template set.
