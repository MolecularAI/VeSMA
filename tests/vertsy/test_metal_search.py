"""End-to-end retrosynthesis search on the Metal backend.

Drives the full search loop (init -> search -> step -> keyed_match -> intern ->
expansions -> iter_solved_routes) with `MetalFoldedPrefilteredReactionMatcher` and checks
that Metal recovers the same route as the CPU backend on a small solvable target.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from vertsy.expansions import Expansions
from vertsy.interner import DictInterner
from vertsy.routes import iter_solved_routes
from vertsy.search import expansion, init, object_mol_array, search
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.interop import cmol_from_mol
from vesma.keyer import short_ring_wl_keyer
from vesma.reaction import parse_reaction_smarts

pytestmark = [pytest.mark.gpu, pytest.mark.metal]


def skip_if_no_metal_toolchain() -> None:
    from vesma.metal_matcher import GpuUnavailableError, probe_metal

    try:
        probe_metal()
    except (GpuUnavailableError, RuntimeError) as exc:
        pytest.skip(f"Metal backend not available: {exc}")


def _run_search(match_fn, target_keys, frontier_mols, stock_keys, max_depth):
    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    return search(
        target_keys,
        frontier_mols,
        stock_keys,
        max_depth,
        expansions,
        match=match_fn,
    )


def test_metal_search_recovers_route_and_agrees_with_cpu() -> None:
    skip_if_no_metal_toolchain()
    from vesma.matcher import all_vs_all, keyed_match
    from vesma.metal_reaction_matcher import (
        MetalFoldedPrefilteredReactionMatcher,
        MetalReactionMatcher,
        metal_short_ring_wl_key,
        metal_stack_mols,
    )

    rules = [parse_reaction_smarts("[C:1][C:2]>>[C:1].[C:2]")]
    target = "CC"  # ethane -> two methanes, both in stock
    codec = short_ring_wl_keyer
    stock_keys = {codec.from_smiles("C")}
    max_depth = 1

    # Metal
    metal_matcher = MetalFoldedPrefilteredReactionMatcher(MetalReactionMatcher(rules))
    target_keys_m, frontier_m = init(
        [target],
        from_smiles_mol=lambda s: cmol_from_mol(Chem.MolFromSmiles(s)),
        from_smiles_key=codec.from_smiles,
        stack_mols=metal_stack_mols,
    )
    metal_exp = _run_search(
        expansion(keyed_match(all_vs_all(metal_matcher), metal_short_ring_wl_key)),
        target_keys_m,
        frontier_m,
        stock_keys,
        max_depth,
    )
    metal_routes = frozenset(
        iter_solved_routes(target_keys_m[0], metal_exp, stock_keys)
    )

    # CPU
    cpu_matcher = CpuReactionMatcher(rules)

    def cpu_match(mols):
        precursors, hits = cpu_matcher.match(mols)
        keys = [codec.encode(p) for p in precursors]
        return object_mol_array(precursors), keys, hits

    target_keys_c, frontier_c = init(
        [target],
        from_smiles_mol=lambda s: cmol_from_mol(Chem.MolFromSmiles(s)),
        from_smiles_key=codec.from_smiles,
    )
    cpu_exp = _run_search(
        expansion(cpu_match), target_keys_c, frontier_c, stock_keys, max_depth
    )
    cpu_routes = frozenset(iter_solved_routes(target_keys_c[0], cpu_exp, stock_keys))

    assert metal_routes, "Metal failed to recover the ethane -> 2x methane route."
    assert metal_routes == cpu_routes


def test_aostar_runs_on_the_metal_backend() -> None:
    """AO* reaches a GPU backend at all — the property the batch-space frontier buys.

    Nothing here is about speed. Until the frontier was carried as a `(sids, batch)` pair,
    AO* held molecules in a per-sid dict and had to index one out of a batch, which a packed
    Metal batch refuses by design ("molecule batch selection must preserve the batch axis")
    and which no decoder could satisfy anyway. So this test could not have been written.

    `zero_heuristic` because it is the only heuristic that works on any `M`: scoring a packed
    batch needs a Metal-side heuristic, which is a separate piece of work. `short_ring_wl_keyer`
    keys are tuples, not strings, so this also pins that the search never assumes an invertible
    key space.
    """
    skip_if_no_metal_toolchain()
    from vertsy.aostar import aostar, constant_step
    from vertsy.heuristics import zero_heuristic
    from vertsy.routes import reaction_count
    from vesma.matcher import all_vs_all, keyed_match
    from vesma.metal_reaction_matcher import (
        MetalReactionMatcher,
        metal_concat_batches,
        metal_short_ring_wl_key,
        metal_stack_mols,
    )

    codec = short_ring_wl_keyer
    matcher = MetalReactionMatcher([parse_reaction_smarts("[C:1][C:2]>>[C:1].[C:2]")])
    target = "CCC"  # propane -> methanes, two reactions
    stock = frozenset({codec.from_smiles("C")})

    routes = aostar(
        codec.from_smiles(target),
        price=constant_step(1.0),
        target_mol=cmol_from_mol(Chem.MolFromSmiles(target)),
        stock=stock,
        expansions=Expansions(NumpyExpansionStore(), DictInterner()),
        heuristic=zero_heuristic,
        match=expansion(keyed_match(all_vs_all(matcher), metal_short_ring_wl_key)),
        stack_mols=metal_stack_mols,
        concat_mols=metal_concat_batches,
        width=8,
        passes=3,
    )

    assert routes, "AO* must solve propane on Metal"
    assert min(reaction_count(route) for route in routes) == 2


def test_the_metal_frontier_refuses_an_oversized_molecule_rather_than_crashing() -> (
    None
):
    """A molecule over the bucket is a refusal, not a numpy broadcast error.

    Every packed structure on this arm is `_BUCKET` atom slots wide. An oversized *precursor* has
    always been refused and counted, but the frontier path packed whatever it was given, so a
    target larger than the bucket reached `_pack_atom_masks` and raised `could not broadcast
    input array from shape (68,) into shape (64,)` from inside numpy -- taking a whole sweep with
    it rather than costing one target. PaRoutes never reaches the cap (n1 tops out at 43 atoms),
    which is why it went unnoticed; Pistachio-Hard has targets of 65, 68 and 85.
    """
    from vesma import cmol_from_smiles
    from vesma.metal_reaction_matcher import _BUCKET, metal_stack_mols
    from vesma.rejection import ChemistryRejection, RejectionReason

    # A straight chain is the shortest way to say "more atoms than the bucket".
    oversized, _ = cmol_from_smiles("C" * (_BUCKET + 1))
    assert len(oversized.Z) == _BUCKET + 1

    with pytest.raises(ChemistryRejection) as refused:
        metal_stack_mols([oversized])
    assert refused.value.reason == RejectionReason.PRECURSOR_CAPACITY_EXCEEDED
    assert str(_BUCKET + 1) in str(refused.value)

    # One atom smaller still packs, so the guard is at the bucket and not below it.
    fits, _ = cmol_from_smiles("C" * _BUCKET)
    assert metal_stack_mols([fits]).atom_counts.tolist() == [_BUCKET]
