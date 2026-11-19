ChemoDOTS Enamine building-block stock.

Layout:
- `processed/chemdots_enamine_bbs.smi.bz2`: derived SMILES-only stock file used
  for paper-scale retrosynthesis throughput benchmarks. Committed.
- `processed/chemdots_enamine_bbs_annotated.smi.bz2`: the same catalogue keeping
  the Enamine `name` column. Gitignored and re-fetchable, because it is an order
  of magnitude larger and only a *report* needs it.
- `raw/`: gitignored landing spot for the full upstream download (not vendored).
- `prepare.sh`: downloads the source table once and writes both derived files.

The two are interchangeable for searching and share one prepared-stock cache, so
adding the annotated file re-keys nothing. `open_smiles` keeps the first
whitespace-delimited column, and the cache is keyed on the resulting tuple of
SMILES rather than on a path -- the annotation never reaches the key. Asserted by
`test_an_annotated_catalogue_shares_its_stripped_sibling_s_cache`. Use the stock
name `enamine` for a fresh checkout and `enamine-annotated` once downloaded; the
only difference is whether a route's starting materials can print the catalogue
number a chemist orders by.

Attribution and licensing:
- Source data: ChemoDOTS web server data files, Hoffer L.,
  Charifi-Hoareau G., Barelier S., Betzi S., Miller T. W., Morelli X.,
  Roche P., Zenodo, `https://doi.org/10.5281/zenodo.10776787`.
- Source file URL:
  `https://zenodo.org/records/10776787/files/ChemoDOTS_Enamine_BBs.smi?download=1`
- The ChemoDOTS publication states that the final building blocks from MolPort
  and Enamine are available in SMILES format from the Zenodo repository.
- The Zenodo record is published under Creative Commons Attribution 4.0
  International.

Derived manifest policy:
- The source file is a tab-delimited table with header `smiles	name`.
- The committed derived file keeps only the first column (one SMILES per line),
  dropping the header and the Enamine catalog `name` column. The gitignored
  annotated file keeps both columns and drops only the header.
- No canonicalization, chemistry cleanup, parser-compatibility filtering, or
  deduplication is performed during this conversion.
- The derived stock is intended only for benchmark stock-membership checks.

Counts and checksums:
- Source access date: 2026-06-03.
- Source raw bytes: `41,293,833`.
- Source raw rows: `988,112` molecules plus header.
- Source raw SHA256:
  `18f8112b99205de9b62046c3dbbc7ff96c4d308f3c14f5398908d73b09565afd`
- Derived decompressed rows: `988,112`.
- Derived decompressed SHA256:
  `5ece482b17fd0dd077cdce83bea9aa1cc124b28669630c01fc6d24a2268cf38b`
- Derived compressed bytes: `4,902,868`.
- Derived compressed SHA256:
  `f6c5d0fc4c24afa66f931c1d4c0636f437570a218eb10745863aafcc8bac12b0`

Regeneration:

```bash
bash tests/data/external/chemdots-stock/prepare.sh
```
