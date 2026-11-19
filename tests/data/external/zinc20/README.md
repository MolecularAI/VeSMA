# ZINC20 purchasable stock

`prepare.sh` downloads a purchasable ZINC20 stock as SMILES, per tranche, from
`files.docking.org`. **It has not been run yet** -- it is here so the cost is known and the fetch
is one command when a ZINC-scale stock is wanted.

This directory exists because the ZINC stock AiZynthFinder distributes cannot be used here: it is
17.4M InChI keys with no structures, and an InChI key is a one-way hash. See
`../aizynthfinder/README.md`. ZINC20's own tranche catalogue is SMILES, which is the route that
leaves the search unchanged.

## Layout

```
https://files.docking.org/2D/<TT>/<TT><r><p>.smi
```

`TT` is the molecular-weight and logP tranche, `AA` through `KK`, 121 of them. `r` is reactivity,
one of `A B C E G I`. `p` is purchasability, `A` through `F`. Each file is `smiles zinc_id`,
space-delimited, with a one-line header. The cross product is not full -- a sparse corner of the
property space has fewer files -- so `prepare.sh` reads each tranche's directory listing rather
than generating names.

## Cost, measured

Apache's directory listings carry file sizes, so the whole catalogue can be measured with 121
small requests and no payload. As of the 2025-01-28 build:

| purchasability | files | size |
| --- | --- | --- |
| `A` | 715 | 0.16 GB |
| `B` | 717 | 0.45 GB |
| `C` | 694 | 0.01 GB |
| `D` | 712 | 50.35 GB |
| `E` | 722 | 55.37 GB |
| `F` | 715 | 0.53 GB |

`D` and `E` are make-on-demand space -- the billions ZINC20 can enumerate and nobody holds. The
other four are the catalogue that physically exists: **2,841 files, 1.14 GB** uncompressed, about
330 MB gzipped and on the order of 23M molecules. That last figure is the reason to believe the
split is read correctly: AiZynthFinder's ZINC stock is 17.4M rows, the same order.

Time is latency-bound rather than bandwidth-bound -- 2,841 requests at six concurrent connections,
so tens of minutes rather than the minute the bytes alone would take.

`PURCHASABILITY` defaults to `ABCF` and can be narrowed; `A` alone is 715 files and 0.16 GB.
`TRANCHES` restricts the grid, for a smoke test. `JOBS` sets concurrency, six by default, which is
polite to a shared academic host.

**The letter-to-label mapping is not asserted here, and is an open question.** The script excludes
`D` and `E` because they are two orders of magnitude larger than the rest, not because a document
says which code means "in-stock" and which means "agent". The server carries no legend --
`files.docking.org/2D/README` and `/README` are both 404 -- so settling it means reading ZINC20's
tranche browser at `https://zinc20.docking.org/tranches/home`, which has not been done.

That matters because of the intended shape of the result: **two stocks, not one.** A narrow one, if
`A` turns out to mean better-stocked or faster-delivered, and the full non-make-on-demand union
`ABCF`. A solvability ceiling measured against both says how much of the answer depends on how
patient the chemist is, which one stock cannot say. Until the labels are checked, `A` is only "the
smallest of the four physical columns", which is not a reason to prefer it.

Reactivity is deliberately *not* filtered. An acid chloride is "reactive" to a docking screen and
a building block to a chemist, and a retrosynthesis stock wants it.

## Output

```
processed/zinc20-in-stock.smi.gz            SMILES only. What a search loads.
processed/zinc20-in-stock-annotated.smi.gz  SMILES and the ZINC id, for a report.
```

Both gitignored, both written from one download, deduplicated on the SMILES with the first ZINC id
kept. Same pair convention as `../chemdots-stock`, and for the same reason: `open_smiles` keeps the
first whitespace-delimited column, so the two read as the same tuple of SMILES and share one
prepared-stock cache.

Nothing in this directory is committed but the script and these notes. `raw/` and `processed/` are
gitignored and the download is re-fetchable.

## Attribution and licensing

- ZINC20: Irwin J. J., Tang K. G., Young J., Dandarchuluun C., Wong B. R., Khurelbaatar M.,
  Moroz Y. S., Mayfield J., Sayle R. A., *J. Chem. Inf. Model.* 60, 6065-6073 (2020),
  `https://doi.org/10.1021/acs.jcim.0c00675`.
- ZINC: Irwin J. J., Shoichet B. K., *J. Chem. Inf. Model.* 45, 177-182 (2005).
- ZINC is free for everyone to use. The per-tranche files are served without a licence file; the
  database's own terms are at `https://zinc20.docking.org/`. Nothing from it is redistributed from
  this repository.
