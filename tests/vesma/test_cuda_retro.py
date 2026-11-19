from __future__ import annotations

import numpy as np
import pytest

from vesma import CMol, cmol_from_smiles, gmol_from_cmol
from vesma.bonds import (
    BOND_KIND_MASK,
    BOND_TOPOLOGY_ANY,
    BOND_TOPOLOGY_MASK,
)
from vesma.cuda_prune import match_fused_assignments_cuda, prepare_cuda_prune_batch
from vesma.cuda_retro import (
    CudaMoleculeStore,
    construct_retro_cuda,
    prepare_cuda_molecule_batch,
    prepare_cuda_reaction_batch,
    wl_hash_kernel,
)
from vesma.identity import short_ring_wl_molecule_id
from vesma.limits import CUDA_EDITED_ATOM_CAP
from vesma.reaction import parse_reaction_smarts, precursors_from_mapping
from vesma.rejection import ChemistryRejection, RejectionReason
from vesma.rings import RingDescriptors


def _skip_if_no_cuda() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")


def _cuda_outcomes(
    *,
    product: CMol,
    reaction_smarts: str,
) -> list[frozenset[tuple[int, int]] | RejectionReason]:
    reaction = parse_reaction_smarts(reaction_smarts)
    batch = prepare_cuda_prune_batch([reaction.lhs], [gmol_from_cmol(product)])
    match = match_fused_assignments_cuda(
        batch,
        max_leaf_checks=4096,
    )
    retro = construct_retro_cuda(
        batch,
        match,
        prepare_cuda_reaction_batch([reaction]),
    )
    lo = retro.precursor_hash_lo_host()
    hi = retro.precursor_hash_hi_host()
    assignment_ids = retro.precursor_assignment_ids_host()
    grouped: dict[int, set[tuple[int, int]]] = {}
    for assignment_id, hash_lo, hash_hi in zip(assignment_ids, lo, hi, strict=True):
        grouped.setdefault(int(assignment_id), set()).add((int(hash_lo), int(hash_hi)))
    reasons = retro.rejection_reasons_host()
    return [
        (
            RejectionReason(int(reason))
            if reason != np.uint32(0xFFFFFFFF)
            else frozenset(grouped.get(assignment_id, set()))
        )
        for assignment_id, reason in enumerate(reasons)
    ]


def _cpu_outcomes(
    *,
    product: CMol,
    reaction_smarts: str,
) -> list[frozenset[tuple[int, int]] | RejectionReason]:
    reaction = parse_reaction_smarts(reaction_smarts)
    batch = prepare_cuda_prune_batch([reaction.lhs], [gmol_from_cmol(product)])
    match = match_fused_assignments_cuda(
        batch,
        max_leaf_checks=4096,
    )
    count = int(match.assignment_counts_host()[0, 0])
    assignments = match.assignments_host()[:count, : reaction.lhs.atom_count]
    outcomes: list[frozenset[tuple[int, int]] | RejectionReason] = []
    for mapping in assignments:
        try:
            precursors = precursors_from_mapping(
                product=product,
                reaction=reaction,
                mapping=mapping,
                ring_descriptors=RingDescriptors.CONNECTIVITY,
            )
        except ChemistryRejection as rejection:
            outcomes.append(rejection.reason)
            continue
        outcomes.append(
            frozenset(short_ring_wl_molecule_id(precursor) for precursor in precursors)
        )
    return outcomes


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("product_smiles", "reaction_smarts"),
    [
        ("CC", "[C:1][C:2]>>[C:1].[C:2]"),
        ("COC", "[C:1][O][C:2]>>[C:1]-[C:2]"),
        ("C", "[C:1]>>[C:1]-O"),
        ("C", "[C:1]>>[C:1].[O-]"),
        ("C", "[C:1]>>[OH-]"),
        ("N", "[N:1]>>[NH2:1]"),
        ("[NH4+]", "[N+:1]>>[N:1]"),  # omitted charge is inherited
        ("[NH4+]", "[N+:1]>>[N;H3;+0:1]"),  # explicit neutral charge
        ("[NH4+]", "[N+:1]>>[N;+0:1]"),  # charge change requires written H
        ("C", "[C:1]>>[C:1].[NH2]"),  # one bad component rejects the application
        # Two components fail for different reasons in one application. The first
        # component in CPU construction order wins; enum order must not replace it.
        (
            "C=CCC1(C)CN(C)Cc2c1n(CC(=C)c1ccc(F)cc1)c1ccc(C)cc21",
            "[c;H0;D3;+0:5]1:[c;H0;D3;+0:4]:[n;H0;D3;+0:3]:"
            "[c;H0;D3;+0:2]:[c;H0;D3;+0:1]:1>>"
            "Cl-[c;H0;D3;+0:1]:[c;H0;D3;+0:2]-[NH;D2;+0:3]."
            "O=[C;H0;D3;+0:4]-[CH2;D2;+0:5]",
        ),
        ("C1=CC1", "[C:1]1[C:2][C:3]1>>[C:1]1=[C:2][C:3]1"),
        ("C1=CCC1", "[C:1]1[C:2][C:3][C:4]1>>[C:1]1=[C:2][C:3][C:4]1"),
        ("C", "C>>O"),
        # Heteroatoms across their valence options: the unified valence/H model
        # must infer the same precursor hydrogen counts on CPU and device.
        ("CS", "[C:1][S:2]>>[C:1].[S:2]"),  # S(2)
        ("CP", "[C:1][P:2]>>[C:1].[P:2]"),  # P(3)
        ("CB", "[C:1][B:2]>>[C:1].[B:2]"),  # B(3)
        ("CCl", "[C:1][Cl:2]>>[C:1].[Cl:2]"),  # Cl(1)
        ("CF", "[C:1][F:2]>>[C:1].[F:2]"),  # F(1)
        ("CS(C)(=O)=O", "[C:1][S:2]>>[C:1].[S:2]"),  # S(6)
        ("OP(O)O", "[O:1][P:2]>>[O:1].[P:2]"),  # P(3) with O survivors
        # Aromaticity re-perception: breaking an aromatic ring leaves bonds that
        # still carry aromatic bits, so the WL H must be re-derived from the
        # canonical bonds (input-invariant), not read from the stored perception H.
        ("c1ccccc1", "[c:1][c:2]>>[c:1].[c:2]"),  # break benzene
        ("c1ccncc1", "[c:1][c:2]>>[c:1].[c:2]"),  # break pyridine
        ("c1cc[nH]c1", "[c:1][c:2]>>[c:1].[c:2]"),  # break pyrrole
        ("c1ccoc1", "[c:1][c:2]>>[c:1].[c:2]"),  # break furan
        ("c1ccc2ccccc2c1", "[c:1][c:2]>>[c:1].[c:2]"),  # naphthalene (fused)
        ("c1ccccc1C", "[c:1][C:2]>>[c:1].[C:2]"),  # detach methyl, ring intact
    ],
)
def test_construct_retro_cuda_matches_cpu_short_ring_wl_tokens(
    product_smiles: str,
    reaction_smarts: str,
) -> None:
    _skip_if_no_cuda()

    product = cmol_from_smiles(product_smiles)[0]
    observed = _cuda_outcomes(
        product=product,
        reaction_smarts=reaction_smarts,
    )
    expected = _cpu_outcomes(
        product=product,
        reaction_smarts=reaction_smarts,
    )

    assert observed == expected


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize(
    "smiles",
    [
        "C",
        "C" * 30,
        "C" * 46,
        "C" * 64,
        "C1CC1",
        "C1CCCCCCC1",
        "C1CCC(CC1)C2CCC(CC2)C3CCC(CC3)C4CCC(CC4)",
        "CC.CCC.c1ccccc1",
        "[CH-]1C=CC=C1",  # cyclopentadienyl anion: charged aromatic ring carbon (H=1)
        "[cH+]1cccccc1",  # tropylium: charged aromatic ring carbon (H=1)
    ],
)
def test_warp_wl_hash_matches_cpu_at_size_and_ring_boundaries(smiles: str) -> None:
    _skip_if_no_cuda()
    molecule = cmol_from_smiles(smiles)[0]
    batch = prepare_cuda_molecule_batch([((0, 0), gmol_from_cmol(molecule))])
    lo, hi = wl_hash_kernel(batch)

    assert (int(lo.get()[0]), int(hi.get()[0])) == short_ring_wl_molecule_id(molecule)


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize(
    "smiles",
    [
        "CCCCCC",  # chain: no ring edges
        "C1CC1",  # 3-ring
        "C1CCC1",  # 4-ring
        "C1CCCC1",  # 5-ring
        "C1CCCCC1",  # 6-ring
        "C1CCCCCC1",  # 7-ring
        "C1CCCCCCC1",  # 8-ring (inclusive upper bound)
        "C1CCCCCCCC1",  # 9-ring: no FLEX edges
        "C1CCCCCCCCCCC1",  # 12-ring: no FLEX edges
        "c1ccccc1",  # aromatic benzene
        "C1=CC=CC=C1",  # Kekule benzene
        "C1=CC2=CC=CC=C2C=C1",  # naphthalene (fused, shared bond)
        "C1CC2CCC1C2",  # norbornane (bridged bicyclic)
        "C1CCC2CCCCC2C1",  # decalin (fused 6-6)
        "O1CCNCC1",  # morpholine (hetero 6-ring)
        "C1CC1C1CC1",  # two cyclopropanes joined by a bond
    ],
)
def test_device_short_cycle_matches_cpu_edge_mask(smiles: str) -> None:
    """GPU bond_in_short_cycle marks the identical 3-8 ring-edge set as CPU
    short_cycle_edge_mask. Compiles the exact production device function so the
    two idiomatic formulations (per-edge alternate-path search vs full-cycle
    enumeration) cannot diverge on ring-size boundaries or fused/bridged rings."""
    _skip_if_no_cuda()
    import cupy as cp

    from vesma.cuda_retro import _DEVICE_SHORT_CYCLE_SOURCE
    from vesma.rings import short_cycle_edge_mask

    probe_source = (
        "#define CUDA_BUCKET_SIZE 64\n#define BOND_NONE 0\n"
        + _DEVICE_SHORT_CYCLE_SOURCE
        + r"""
extern "C" __global__ void probe_short_cycle(
    const unsigned char* bonds_flat,
    const int* comp_atoms,
    int comp_count,
    unsigned char* out
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= comp_count * comp_count) return;
    const int left = idx / comp_count;
    const int right = idx % comp_count;
    unsigned char (*bonds)[CUDA_BUCKET_SIZE] =
        (unsigned char(*)[CUDA_BUCKET_SIZE])bonds_flat;
    const bool present = bonds[comp_atoms[left]][comp_atoms[right]] != BOND_NONE;
    out[idx] = (present && bond_in_short_cycle(
        bonds, comp_atoms, comp_count, left, right)) ? 1 : 0;
}
"""
    )
    kernel = cp.RawModule(code=probe_source, options=("-std=c++11",)).get_function(
        "probe_short_cycle"
    )

    bonds = cmol_from_smiles(smiles)[0].bonds
    n = bonds.shape[0]
    padded = np.zeros((64, 64), dtype=np.uint8)
    padded[:n, :n] = bonds
    comp_atoms = np.arange(n, dtype=np.int32)
    out = cp.zeros(n * n, dtype=cp.uint8)
    threads = 128
    blocks = (n * n + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            cp.asarray(padded.reshape(-1)),
            cp.asarray(comp_atoms),
            np.int32(n),
            out,
        ),
    )
    device = out.get().reshape(n, n).astype(bool)
    expected = short_cycle_edge_mask(bonds != 0)
    assert np.array_equal(device, expected)


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_molecule_store_reuses_constructed_components() -> None:
    _skip_if_no_cuda()

    product = cmol_from_smiles("CC")[0]
    product_token = short_ring_wl_molecule_id(product)
    reaction = parse_reaction_smarts("[C:1][C:2]>>[C:1].[C:2]")
    batch = prepare_cuda_prune_batch([reaction.lhs], [gmol_from_cmol(product)])
    match = match_fused_assignments_cuda(batch)
    retro = construct_retro_cuda(
        batch,
        match,
        prepare_cuda_reaction_batch([reaction]),
    )

    store = CudaMoleculeStore.empty()
    store.add_host_molecules(((product_token, gmol_from_cmol(product)),))
    precursor_batch = store.add_cuda_precursors(retro)

    carbon = short_ring_wl_molecule_id(cmol_from_smiles("C")[0])
    assert carbon in store.tokens
    assert store.tokens == {product_token, carbon}
    assert precursor_batch is not None
    assert precursor_batch.tokens == (carbon,)
    assert precursor_batch.count == 1


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_molecule_store_marks_generated_bonds_topology_any() -> None:
    _skip_if_no_cuda()

    product = cmol_from_smiles("CC")[0]
    reaction = parse_reaction_smarts("[C:1][C:2]>>[C:1][C:2]")
    batch = prepare_cuda_prune_batch([reaction.lhs], [gmol_from_cmol(product)])
    match = match_fused_assignments_cuda(batch)
    retro = construct_retro_cuda(
        batch,
        match,
        prepare_cuda_reaction_batch([reaction]),
    )

    precursor_batch = CudaMoleculeStore.empty().add_cuda_precursors(retro)
    assert precursor_batch is not None
    bonds = precursor_batch.bonds_device.get()
    classes = bonds & np.uint8(BOND_KIND_MASK)
    present = classes != 0
    topology = bonds & np.uint8(BOND_TOPOLOGY_MASK)
    # Perception decides ring or chain per edge; neither "either" nor "neither" is a
    # state a perceived precursor may leave behind.
    assert np.all(topology[present] != BOND_TOPOLOGY_ANY)
    assert np.all(topology[present] != 0)
    assert np.all(bonds[~present] == 0)


@pytest.mark.gpu
@pytest.mark.cuda
def test_construct_retro_cuda_retries_precursor_capacity_overflow() -> None:
    _skip_if_no_cuda()

    product = cmol_from_smiles("CCC")[0]
    reaction = parse_reaction_smarts("[C:1][C:2]>>[C:1].[C:2]")
    batch = prepare_cuda_prune_batch([reaction.lhs], [gmol_from_cmol(product)])
    match = match_fused_assignments_cuda(batch)

    retro = construct_retro_cuda(
        batch,
        match,
        prepare_cuda_reaction_batch([reaction]),
        compute_wl=False,
        precursor_capacity=1,
    )

    assert retro.precursor_required == 8
    assert retro.precursor_capacity == 8
    assert retro.overflow_retries == 1
    assert len(retro.precursor_assignment_ids_host()) == 8
    assert len(retro.precursor_atom_counts_host()) == 8


@pytest.mark.gpu
@pytest.mark.cuda
def test_construct_retro_cuda_refuses_application_with_oversize_component() -> None:
    """One precursor over the cap refuses the whole application, not just that piece.

    The water co-precursor fits and would once have been returned on its own. A reaction
    is defined by all of its named precursors, so a partial set presents an impossible
    disconnection as a complete one. The CPU reference and Metal apply the same rule.
    """
    _skip_if_no_cuda()

    product = cmol_from_smiles("C")[0]
    large_component = "[C:1]" + "C" * 64
    reaction = parse_reaction_smarts(f"[C:1]>>{large_component}.O")
    batch = prepare_cuda_prune_batch([reaction.lhs], [gmol_from_cmol(product)])
    match = match_fused_assignments_cuda(batch)

    with pytest.warns(UserWarning, match="over 64 atoms"):
        retro = construct_retro_cuda(
            batch,
            match,
            prepare_cuda_reaction_batch([reaction]),
        )

    assert retro.precursor_required == 0
    assert retro.precursor_atom_overflow_components == 1
    assert retro.precursor_atom_overflow_assignments == 1


@pytest.mark.gpu
@pytest.mark.cuda
def test_construct_retro_cuda_drops_oversize_edited_assignment() -> None:
    _skip_if_no_cuda()

    product = cmol_from_smiles("C" * 64)[0]
    added_atoms = ".".join("O" for _ in range(CUDA_EDITED_ATOM_CAP - 63))
    reaction = parse_reaction_smarts(f"[C:1]>>[C:1].{added_atoms}")
    batch = prepare_cuda_prune_batch([reaction.lhs], [gmol_from_cmol(product)])
    match = match_fused_assignments_cuda(batch)

    with pytest.warns(UserWarning, match=f"over {CUDA_EDITED_ATOM_CAP} edited atoms"):
        retro = construct_retro_cuda(
            batch,
            match,
            prepare_cuda_reaction_batch([reaction]),
        )

    assert match.assignment_required > 0
    assert retro.precursor_required == 0
    assert retro.edited_atom_overflow_assignments == match.assignment_required


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_molecule_batch_getitem_device_gather() -> None:
    """CudaMoleculeBatch.__getitem__ gathers a subset on-device (bool / int / slice),
    parallel to a WL re-hash of the selected rows."""
    _skip_if_no_cuda()
    from vesma.cuda_retro import (
        CudaMoleculeBatch,
        prepare_cuda_molecule_batch,
        wl_hash_kernel,
    )

    smiles = ["CCO", "CCN", "CCCC", "c1ccccc1"]
    batch = prepare_cuda_molecule_batch(
        [((i, 0), gmol_from_cmol(cmol_from_smiles(s)[0])) for i, s in enumerate(smiles)]
    )
    counts = batch.atom_counts_device.get().tolist()
    lo, _ = wl_hash_kernel(batch)
    lo = lo.get()

    # Integer index array (order preserved).
    sub = batch[np.array([2, 0])]
    assert sub.count == 2
    assert sub.tokens == ((2, 0), (0, 0))
    assert sub.atom_counts_device.get().tolist() == [counts[2], counts[0]]
    sub_lo, _ = wl_hash_kernel(sub)
    assert sub_lo.get().tolist() == [int(lo[2]), int(lo[0])]

    # Boolean mask.
    masked = batch[np.array([True, False, True, False])]
    assert masked.tokens == ((0, 0), (2, 0))

    # Slice.
    assert batch[1:3].tokens == ((1, 0), (2, 0))

    # Empty selection -> empty batch.
    assert batch[np.array([], dtype=int)].count == 0

    # tokens=() (de-fused-keyer precursor batch): gather keeps tokens empty, count
    # falls back to the atom axis.
    untokened = CudaMoleculeBatch(
        tokens=(),
        atom_counts_device=batch.atom_counts_device,
        atoms_device=batch.atoms_device,
        bonds_device=batch.bonds_device,
        rows_device=batch.rows_device,
    )
    gathered = untokened[np.array([3, 1])]
    assert gathered.tokens == ()
    assert gathered.count == 2
    assert gathered.atom_counts_device.get().tolist() == [counts[3], counts[1]]

    # from_parts round-trips a split-then-rejoin.
    rejoined = CudaMoleculeBatch.from_parts([batch[0:2], batch[2:4]])
    assert rejoined is not None
    assert rejoined.atom_counts_device.get().tolist() == counts

    with pytest.raises(TypeError, match="preserve the batch axis"):
        batch[0]
    with pytest.raises(IndexError, match="one-dimensional"):
        batch[np.array([[0, 1]])]
    with pytest.raises(TypeError, match="booleans or integers"):
        batch[np.array([0.0])]
    with pytest.raises(IndexError, match="out of range"):
        batch[np.array([len(smiles)])]
