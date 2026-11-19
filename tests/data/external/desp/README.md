# DESP benchmark target sets

Target sets from *Double-ended synthesis planning with goal-constrained bidirectional search*,
Yu K., Roh J., Li Z., Gao W., Wang R., Coley C. W., arXiv:2407.06334 (2024), used here as hard
target lists.

- `prepare.sh` fetches `benchmarks.zip` (11 kB) from the paper's figshare record and writes, per
  set, a `*-targets.smi` of target SMILES and a `*-pairs.tsv` keeping the starting material.
- Cases: `pistachio_hard` 100, `pistachio_reachable` 150, `uspto_190` 190.
- Nothing is committed. The target SMILES derive from Pistachio, a commercial NextMove database,
  so `raw/` and `processed/` are gitignored and re-fetched rather than vendored. The 10.1 GB
  `desp_data.zip` from the same record is model weights and is not wanted here.

## These are double-ended cases, and that changes what a number means

Each upstream line is a tuple `('<target>', '<starting material>')`. DESP searches retro from the
target *and* forward from that starting material, and a case counts as solved by a route joining
the two. **Vesma's search is single-ended to a stock.** Running `*-targets.smi` and ignoring the
starting materials is therefore a *different and easier task*: any route to any stocked material
counts, where DESP must arrive at one specified molecule.

So a solve rate measured here is not comparable with the paper's, in either direction, and must
not be reported as though it were. What it is good for is what the sets were selected to be: hard
targets. `*-pairs.tsv` keeps the starting materials so the constrained task stays available if
goal-constrained search is ever built.

## None of DESP's strategies is a rule-based baseline

Worth recording, since it is the obvious question to ask of the repo. All six strategies
(`f2e`, `f2f`, `retro`, `retro_sd`, `random`, `bfs`) expand with the same learned
template-relevance MLP (`desp/inference/retro_predictor.py`) and differ only in node selection:
`f2e`/`f2f`/`retro_sd` take `min(total_value + min(total_distance))` from a learned value model
and a learned synthetic-distance model; `bfs` takes the lowest-depth open node with ties broken by
lowest parent reaction cost; `random` picks uniformly. And `cost = -1 * np.log(score)`
(`desp/search/data_structures/desp_tree.py:100`) is that same model's softmax probability -- so
`bfs` is "no learned *value function*", not "no learned model", and the repo has no
uninformed-expansion baseline to compare a rule-based engine against.

The parts of their scoring that could have been borrowed turn out to be shapes this engine already
uses: `-log(score)` step costs are what `StepPrice` and `cost_of_factor` do, and
`total_value + min(total_distance)` is f = g + h, which is `total = context + inside` with the
`atoms` heuristic standing in for their learned synthetic distance. The genuinely different idea is
the bidirectional search itself, which needs a specified starting material and so is architecture
rather than a heuristic.
