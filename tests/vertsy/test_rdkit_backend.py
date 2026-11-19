from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit.Chem")

from rdkit import Chem

from vertsy.expansions import Expansions, materialize
from vertsy.interner import DictInterner
from vertsy.replay import replay_search
from vertsy.routes import Route, iter_solved_routes, leaves, reaction_depth
from vertsy.search import expansion, init, object_mol_array, search
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma.keyer import (
    rdkit_mol_canonical_smiles,
    rdkit_mol_inchi,
    rdkit_token_from_smiles,
)
from vesma.matcher import all_vs_all, keyed_match
from vesma.rdkit_matcher import (
    RdkitFilterCatalogReactionMatcher,
    RdkitReactionMatcher,
    rdkit_reactions_from_smarts,
)


def _mol_from_smiles(smi: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    return mol


def _key(mols: list[Chem.Mol]) -> list[str]:
    return [rdkit_mol_canonical_smiles(m) for m in mols]


def _reactions(smarts: tuple[str, ...]):
    return rdkit_reactions_from_smarts(smarts)


def test_rdkit_backend_splits_ethane_to_stock_carbon() -> None:
    """The rdkit backend reuses search/build_layer unchanged: ethane → stock carbon."""
    matcher = RdkitReactionMatcher(_reactions(("[C:1][C:2]>>[C:1].[C:2]",)))
    carbon = rdkit_token_from_smiles("C")
    target = rdkit_token_from_smiles("CC")

    frontier_keys, frontier_mols = init(
        ["CC"],
        from_smiles_mol=_mol_from_smiles,
        from_smiles_key=rdkit_token_from_smiles,
    )
    expansions = search(
        frontier_keys,
        frontier_mols,
        {carbon},
        1,
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=expansion(keyed_match(all_vs_all(matcher), _key)),
    )

    # The two RunReactants outcomes collapse to one step in key space.
    assert materialize(expansions) == {target: frozenset({(0, frozenset({carbon}))})}


def test_rdkit_backend_solves_two_step_route() -> None:
    """Propane decomposes to stock carbon over two layers, yielding a solved route."""
    matcher = RdkitReactionMatcher(_reactions(("[C:1][C:2]>>[C:1].[C:2]",)))
    carbon = rdkit_token_from_smiles("C")
    target = rdkit_token_from_smiles("CCC")

    frontier_keys, frontier_mols = init(
        ["CCC"],
        from_smiles_mol=_mol_from_smiles,
        from_smiles_key=rdkit_token_from_smiles,
    )
    expansions = search(
        frontier_keys,
        frontier_mols,
        {carbon},
        2,
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=expansion(keyed_match(all_vs_all(matcher), _key)),
    )

    assert tuple(iter_solved_routes(target, expansions, {carbon}))


def test_rdkit_explicit_pairs_restrict_to_requested_rules() -> None:
    """Explicit pairs match each molecule only against requested rules."""
    rules = {
        0: "[C:1][C:2]>>[C:1].[C:2]",
        1: "[O:1][C:2]>>[O:1].[C:2]",
    }
    matcher = RdkitReactionMatcher(rdkit_reactions_from_smarts(tuple(rules.values())))
    ethane = _mol_from_smiles("CC")
    # Restricting ethane to the C–O rule yields no applications.
    _, hits = matcher.match([ethane], [(0, 1)])
    assert len(hits) == 0

    # Restricting to the C–C rule splits it, and every hit carries that rule id.
    precursors, hits = matcher.match([ethane], [(0, 0)])
    assert {rdkit_mol_canonical_smiles(p) for p in precursors} == {"C"}
    assert len(hits) and all(rule_id == 0 for _, rule_id, _, _ in hits)


def test_rdkit_reaction_matcher_parallel_matches_serial() -> None:
    reactions = _reactions(("[C:1][C:2]>>[C:1].[C:2]",))
    mols = [_mol_from_smiles("CC"), _mol_from_smiles("CCC")]
    pairs = [(0, 0), (1, 0)]

    with RdkitReactionMatcher(reactions, n_jobs=1) as serial:
        serial_precursors, serial_hits = serial.match(mols, pairs)
    with RdkitReactionMatcher(reactions, n_jobs=2) as parallel:
        parallel_precursors, parallel_hits = parallel.match(mols, pairs)

    assert np.array_equal(parallel_hits, serial_hits)
    assert [rdkit_mol_canonical_smiles(mol) for mol in parallel_precursors] == [
        rdkit_mol_canonical_smiles(mol) for mol in serial_precursors
    ]


@pytest.mark.parametrize("key", [rdkit_mol_canonical_smiles, rdkit_mol_inchi])
def test_generic_rdkit_replay_uses_mapped_backend_keys(key) -> None:
    matcher = RdkitReactionMatcher(_reactions(("[C:1][C:2]>>[C:1].[C:2]",)))
    target_key = key(_mol_from_smiles("CC"))
    carbon_key = key(_mol_from_smiles("C"))
    mapped_route = Route(
        target_key,
        0,
        (Route(carbon_key, None), Route(carbon_key, None)),
    )
    stock = leaves(mapped_route)

    def match_fn(mols, pairs):
        precursors, hits = matcher.match(mols, pairs)
        return object_mol_array(precursors), [key(mol) for mol in precursors], hits

    expansions = replay_search(
        mapped_route,
        [target_key],
        object_mol_array([_mol_from_smiles("CC")]),
        stock,
        reaction_depth(mapped_route),
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=match_fn,
    )

    assert mapped_route in frozenset(iter_solved_routes(target_key, expansions, stock))


def test_rdkit_reactions_from_smarts_rejects_invalid_reaction_smarts() -> None:
    with pytest.raises(ValueError, match="RDKit failed to parse reaction SMARTS"):
        rdkit_reactions_from_smarts(("not a reaction",))


# --------------------------------------------------------------- parallelism is invisible

PARALLEL_SMARTS = (
    "[C:1][C:2]>>[C:1].[C:2]",
    "[C:1][O:2]>>[C:1].[O:2]",
    "[C:1][N:2]>>[C:1].[N:2]",
)
PARALLEL_SMILES = ("CCO", "CCN", "CCOCC", "CC(C)N", "OCCN")


def _filter_catalog_matcher(
    n_jobs: int | None = None,
) -> RdkitFilterCatalogReactionMatcher:
    return RdkitFilterCatalogReactionMatcher(
        _reactions(PARALLEL_SMARTS),
        [smarts.split(">>")[0] for smarts in PARALLEL_SMARTS],
        n_jobs=n_jobs,
    )


@pytest.mark.parametrize("batch", [1, 3, 5])
def test_filter_catalog_screen_is_the_same_pairs_at_any_thread_count(
    batch: int,
) -> None:
    """`n_jobs` buys wall clock, never a different answer.

    `RunFilterCatalog` threads the catalog over molecules, so the screen's cost per molecule
    depends on how many are handed to it at once; the *pairs* it finds must not. Batch sizes
    straddle one, where there is no parallelism to be had.
    """
    mols = [_mol_from_smiles(smi) for smi in PARALLEL_SMILES[:batch]]

    with _filter_catalog_matcher(1) as serial, _filter_catalog_matcher(-1) as threaded:
        assert serial.screen(mols) == threaded.screen(mols)


@pytest.mark.parametrize("batch", [1, 3, 5])
def test_rdkit_reaction_matcher_forks_to_the_same_hits(batch: int) -> None:
    """The resident pool must reproduce the serial hits exactly, order included.

    Hit order is not cosmetic: an application's index is assigned by position, and routes are
    keyed by it, so a reordering across `n_jobs` would silently rewrite route identity. Batch
    sizes straddle one, the case a per-call pool could never afford to parallelise.
    """
    reactions = _reactions(PARALLEL_SMARTS)
    mols = [_mol_from_smiles(smi) for smi in PARALLEL_SMILES[:batch]]

    with RdkitReactionMatcher(reactions, n_jobs=1) as serial:
        serial_precursors, serial_hits = serial.match(mols)
    with RdkitReactionMatcher(reactions, n_jobs=-1) as forked:
        forked_precursors, forked_hits = forked.match(mols)

    assert np.array_equal(serial_hits, forked_hits)
    assert [Chem.MolToSmiles(mol) for mol in serial_precursors] == [
        Chem.MolToSmiles(mol) for mol in forked_precursors
    ]


def test_resident_workers_survive_repeated_matches() -> None:
    """The pool is stood up once and reused, and every call gives the same answer.

    The point of resident workers is that call two costs no more than call one, so a matcher
    that is reused must not be quietly rebuilding its pool — nor drifting, since the workers
    carry mutable RDKit reaction objects across calls.
    """
    mols = [_mol_from_smiles(smi) for smi in PARALLEL_SMILES]

    with RdkitReactionMatcher(_reactions(PARALLEL_SMARTS), n_jobs=-1) as matcher:
        first_precursors, first_hits = matcher.match(mols)
        pool = matcher.workers._pool
        assert pool is not None

        for _ in range(3):
            precursors, hits = matcher.match(mols)
            assert np.array_equal(hits, first_hits)
            assert [Chem.MolToSmiles(m) for m in precursors] == [
                Chem.MolToSmiles(m) for m in first_precursors
            ]
        assert matcher.workers._pool is pool

    assert matcher.workers._pool is None


@pytest.mark.parametrize("n_jobs", [1, -1])
def test_in_worker_keying_agrees_with_keying_in_the_parent(n_jobs: int) -> None:
    """Keying in the worker must be indistinguishable from `keyed_match` doing it here.

    The worker additionally collapses duplicate precursors, so the two batches are *not*
    row-for-row identical — the invariant is that they describe the same chemistry: the same
    set of distinct keys, and every hit still resolving to the key it resolved to before.
    """
    reactions = _reactions(PARALLEL_SMARTS)
    mols = [_mol_from_smiles(smi) for smi in PARALLEL_SMILES]

    with RdkitReactionMatcher(reactions, n_jobs=n_jobs) as plain:
        precursors, hits = plain.match(mols)
    parent_keys = _key(list(precursors))

    with RdkitReactionMatcher(reactions, n_jobs=n_jobs, key=_key) as keyed:
        collapsed, worker_keys, collapsed_hits = keyed.match_keyed(mols)

    assert len(worker_keys) == len(collapsed)
    assert set(worker_keys) == set(parent_keys)
    assert len(collapsed) <= len(precursors)

    # Every application still names the same precursor keys it did before collapsing.
    def by_application(hit_rows, keys):
        grouped: dict[tuple[int, int, int], set[str]] = {}
        for mol_idx, rule_id, app_idx, precursor_idx in hit_rows:
            grouped.setdefault((mol_idx, rule_id, app_idx), set()).add(
                keys[precursor_idx]
            )
        return grouped

    assert by_application(collapsed_hits, worker_keys) == by_application(
        hits, parent_keys
    )


def test_worker_collapses_duplicate_precursors() -> None:
    """Near-synonymous templates produce the same precursor; only one row should come home.

    Two rules that disconnect the same bond give identical fragments, so a keyed matcher must
    return fewer precursor rows than hits — the whole point of the 4-column `HitArray`.
    """
    duplicated = ("[C:1][O:2]>>[C:1].[O:2]", "[CX4:1][OX2:2]>>[C:1].[O:2]")
    mols = [_mol_from_smiles("CCO")]

    with RdkitReactionMatcher(_reactions(duplicated), n_jobs=1) as plain:
        _, hits = plain.match(mols)
    with RdkitReactionMatcher(_reactions(duplicated), n_jobs=1, key=_key) as keyed:
        collapsed, keys, collapsed_hits = keyed.match_keyed(mols)

    assert len(hits) == len(collapsed_hits)
    assert len(collapsed) < len(hits)
    assert len(keys) == len(collapsed) == len(set(keys))
    # No orphans: every returned precursor row is named by at least one hit.
    assert {row[3] for row in collapsed_hits} == set(range(len(collapsed)))


def test_closing_a_matcher_is_idempotent_and_it_can_be_used_again() -> None:
    """`close()` releases the workers; a later match transparently spawns a fresh pool.

    Closing twice must not raise, because the `weakref.finalize` backstop may also have run.
    """
    mols = [_mol_from_smiles(smi) for smi in PARALLEL_SMILES]
    matcher = RdkitReactionMatcher(_reactions(PARALLEL_SMARTS), n_jobs=-1)

    _, before = matcher.match(mols)
    matcher.close()
    matcher.close()
    _, after = matcher.match(mols)
    matcher.close()

    assert np.array_equal(before, after)


def test_filter_catalog_screen_rejects_a_molecule_it_cannot_write() -> None:
    """A molecule the screen cannot express reaches the caller as an error, not an empty row.

    Silently screening one out would drop every route through it with no trace. Which of the
    two guards fires is RDKit's business: `MolToSmiles` raises on a molecule it will not write
    (this case), and `RunFilterCatalog` reports a string it will not read as a 'Bad smiles'
    entry instead of raising. Both end as a `ValueError` out of `screen`.
    """
    matcher = _filter_catalog_matcher()
    unwritable = Chem.MolFromSmiles("CN(C)(C)(C)C", sanitize=False)
    assert unwritable is not None

    with pytest.raises(ValueError):
        matcher.screen([unwritable])
