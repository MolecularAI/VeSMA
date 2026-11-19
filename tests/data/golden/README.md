Golden regression snapshots.

These pinned pickles guard the vertsy search/route/replay pipelines against
accidental behavior changes across refactors ("golden" is a slightly strong name —
they are pinned snapshots, not a canonical ground truth).

Files:
- `search_wl_cpu.pkl.gz`: expansion graph, WL-keyed CPU backend.
- `routes_wl_cpu.pkl.gz`: solved routes extracted from that graph.
- `replay_rdkit_cpu.pkl.gz`: restricted replay with RDKit canonical keys.

Regeneration:
- These are produced and consumed by `tests/vertsy/test_golden_search.py`. The
  test writes a golden file if it is absent and otherwise asserts against it.
- To regenerate after an intentional change, delete the relevant `*.pkl.gz` here
  and re-run `pytest tests/vertsy/test_golden_search.py -q`, then commit the new
  snapshot.
