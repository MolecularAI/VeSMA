ChEMBL 50k substructure-search subset.

A small, fixed ChEMBL molecule sample used by perf benchmarks (e.g. matching
ChEMBL molecules against the zbh-smarts PAINS patterns) and as a molecule corpus
for other experiments. We do not vendor full ChEMBL; this 50k subset is enough
for the current benchmarks. A future phase may add the `chembl_downloader`
dependency to fetch and cache full releases on demand.

Layout:
- `raw/chembl16_50K_sss.txt.gz`: unmodified file from Greg Landrum's RDKit blog
  data repo, `https://github.com/greglandrum/rdkit_blog`, file URL
  `https://raw.githubusercontent.com/greglandrum/rdkit_blog/master/data/chembl16_50K_sss.txt.gz`.

Attribution and licensing:
- Redistributed from the RDKit blog repo. The underlying ChEMBL data is made
  available under CC BY-SA 3.0 per the official ChEMBL interface documentation:
  `https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions`.
  ChEMBL's license requires keeping SMILES associated with their ChEMBL ids; the
  raw file is kept intact (it carries both columns) rather than reduced to
  SMILES-only.

Wire format:
- `chembl16_50K_sss.txt.gz`: whitespace-delimited `chembl_id smiles`.
