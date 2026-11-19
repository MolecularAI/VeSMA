# PaRoutes n1-d2 matches against USPTO42k-compatible templates

Derived remix dataset for standalone Vesma match+split scaling benchmarks.

Sources:
- `../../paroutes/processed/n1-d2-f0.smi.gz`: 2,881 PaRoutes n1 depth-2 target
  molecules.
- `../uspto42k-compat/processed/uspto42k-compat.csv.gz`: 42,554 reaction
  SMARTS strings compatible with both RDKit and Vesma.

Artifacts:
- `processed/n1-d2-f0-uspto42k-compat-matches.json.gz`: RDKit
  `FilterCatalog` positive `(molecule_idx, pattern_idx)` pairs for every source
  molecule against the full compatible template set.

The matches file stores only integer pairs plus SHA-256 hashes of the exact
source SMILES and SMARTS tuples. It intentionally does not duplicate molecules
or patterns. Consumers read the source strings, validate the hashes, then filter
pairs by molecule index.

Regenerate with:

```bash
make paroutes-n1-d2-match-uspto42k
```
