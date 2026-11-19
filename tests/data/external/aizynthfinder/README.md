# AiZynthFinder public data

`prepare.sh` downloads the AiZynthFinder public ZINC stock and is **not currently able to derive
a `.smi` from it.** The reason is recorded here because it is a property of the upstream file,
not a problem with the script.

## The ZINC stock is InChI keys, not structures

`https://ndownloader.figshare.com/files/23086469` is `zinc_stock_17_04_20.hdf5`, 633 MB, a pandas
HDF5 table of **17,422,831 rows and one column, `inchi_key`**. There is no SMILES column anywhere
in the file -- `tables.open_file(...).walk_nodes('/')` shows `axis0`, `axis1`, `block0_items` and
`block0_values` and nothing else.

That is correct for AiZynthFinder, whose stock membership test is an InChI-key lookup. It does not
transfer here: vesma keys a molecule by the structure (`short_ring_wl_keyer` on both device arms),
and an InChI key is a one-way hash, so SMILES cannot be recovered from it. Converting this file to
a `.smi` is not possible at any effort.

## What would make it usable, and what each would cost

* **A SMILES-bearing ZINC download.** ZINC20 publishes per-tranche `.smi` from
  `files.docking.org`; the catalogue is many files rather than one, so `prepare.sh` would need a
  tranche list. This is the only option that leaves the search unchanged, and it is the one
  taken: see `../zinc20/`, which reads the tranche listings instead of hardcoding names. The
  purchasable part is 2,841 files and 1.14 GB; the make-on-demand part, excluded, is 106 GB.
* **Keying the search by InChI key instead.** `vesma.keyer` already has
  `inchikey_token_from_smiles` and `rdkit_inchikey_keyer`, so the stock could be loaded
  *pre-keyed* -- the 17.4M keys used directly, with no preparation pass. The cost falls on the
  search side: every generated precursor would need an InChI key, which means an RDKit call per
  molecule where the arms currently hash the structure directly. RDKit is also an optional extra
  here, so such a run would require it.

Nothing is committed from this directory. `raw/` and `processed/` are gitignored; the download is
re-fetchable.

## Attribution and licensing

- AiZynthFinder, Genheden S., Thakkar A., Chadimova V., Reymond J.-L., Engkvist O., Bjerrum E.,
  *J. Cheminform.* 12, 70 (2020), `https://doi.org/10.1186/s13321-020-00472-1`. MIT licensed.
- The stock file is the ZINC subset AiZynthFinder distributes via figshare, download link taken
  from `aizynthfinder/tools/download_public_data.py`.
- ZINC: Irwin J. J., Shoichet B. K., *J. Chem. Inf. Model.* 45, 177-182 (2005).
