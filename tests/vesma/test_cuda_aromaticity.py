from __future__ import annotations

import gzip

import numpy as np
import pytest

from vesma import cmol_from_smiles, perceive_aromaticity
from vesma.identity import identity_seed_from_cmol

_BUCKET = 64


def _skip_if_no_cuda():
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA device is not available.")
    return cp


def _perception_seed(smiles: str):
    """Production identity seed for a molecule, plus the CPU perception reference.
    The device receives canonical seed bonds and the frozen model's molecular-state
    total hydrogen, exactly as precursor perception does in production. The CPU
    reference independently reproduces the CMol aromaticity from the same seed."""
    cmol = cmol_from_smiles(smiles)[0]
    seed = identity_seed_from_cmol(cmol)
    z = seed.Z.astype(np.int16)
    charge = seed.C.astype(np.int8)
    explicit_h = seed.explicit_h.astype(np.int16)
    implicit_h = seed.implicit_h.astype(np.int16)
    bonds = seed.bonds.astype(np.uint8)

    perceived_a, perceived_bonds, _ = perceive_aromaticity(
        Z=z,
        C=charge,
        explicit_h=explicit_h,
        implicit_h=implicit_h,
        bonds=bonds,
    )
    return {
        "z": z,
        "charge": charge,
        "explicit_h": explicit_h,
        "implicit_h": implicit_h,
        "total_h": cmol.H.astype(np.int16),
        "bonds": bonds,
        "perceived_a": perceived_a,
        "perceived_bonds": perceived_bonds,
        "cmol": cmol,
    }


def _run_perceive_batch(seeds):
    from vesma.cuda_retro import probe_cuda_perception

    count = len(seeds)
    z = np.zeros((count, _BUCKET), dtype=np.int32)
    charge = np.zeros((count, _BUCKET), dtype=np.int32)
    total_h = np.zeros((count, _BUCKET), dtype=np.int32)
    bonds = np.zeros((count, _BUCKET, _BUCKET), dtype=np.uint8)
    counts = np.zeros(count, dtype=np.int32)
    for idx, seed in enumerate(seeds):
        n = int(seed["z"].shape[0])
        counts[idx] = n
        z[idx, :n] = seed["z"]
        charge[idx, :n] = seed["charge"]
        total_h[idx, :n] = seed["total_h"]
        bonds[idx, :n, :n] = seed["bonds"]

    return probe_cuda_perception(
        z=z,
        charge=charge,
        total_h=total_h,
        bonds=bonds,
        atom_counts=counts,
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1",  # benzene
        "c1ccncc1",  # pyridine
        "c1cc[nH]c1",  # pyrrole
        "c1ccoc1",  # furan
        "c1ccsc1",  # thiophene
        "c1ccc2ccccc2c1",  # naphthalene (fused 6-6)
        "c1ccc2[nH]ccc2c1",  # indole (fused 5-6 hetero)
        "[CH-]1C=CC=C1",  # cyclopentadienyl anion
        "O=C1C=CC=CN1",  # 2-pyridone
        "C1=CC2=CC=CC=CC2=C1",  # azulene (fused 5-7)
        "[cH+]1cccccc1",  # tropylium (7-ring cation)
        "[o+]1ccccc1",  # pyrylium (O+ as N)
        "[s+]1ccccc1",  # thiopyrylium (S+ as P)
        "O=c1cccccc1",  # tropone
        "C1=CC=CC=CC1",  # cycloheptatriene (not aromatic)
        "C1=CC=CC=CC=C1",  # cyclooctatetraene (not aromatic)
        "C=C1C=CC=C1",  # fulvene (not aromatic)
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "c1ccc(-c2ccccc2)cc1",  # biphenyl (bridge bond not aromatic)
        "Clc1cnc2nc(Cl)c(Cl)nc2c1",  # fused diazine
        # An amine bridging a biaryl's meta positions: the one shape where the atom mask and
        # the circuit's edges came apart, on all three arms at once. The component's joint
        # count passes, the dependent outer cycle is chordless so it is counted as a face,
        # and every edge at the nitrogen then lies on two faces rather than one. The host
        # refused the result outright; the device wrote it into the atom word and read it
        # back two ways. Both arms must now return the nitrogen aliphatic, and identically.
        "N1c2cccc(c2)-c2ccccc21",
        "N1c2ccc(cc2)-c2ccccc21",  # the same bridge one atom longer, para
    ],
)
def test_device_perception_matches_cpu_on_ring_systems(smiles: str) -> None:
    """GPU perceive_aromaticity_device reproduces the CPU atom aromatic mask and
    aromatic bond bits exactly across supported 5-8 ring systems, fused systems,
    aromatic ions, and clearly non-aromatic rings."""
    _skip_if_no_cuda()

    seed = _perception_seed(smiles)
    out_a, out_bonds, status = _run_perceive_batch([seed])
    n = int(seed["z"].shape[0])

    assert int(status[0]) == 0
    # CPU reference reproduces the production CMol aromaticity from the seed.
    assert np.array_equal(seed["perceived_a"], seed["cmol"].A)
    assert np.array_equal(out_a[0, :n], seed["perceived_a"])
    assert np.array_equal(out_bonds[0, :n, :n], seed["perceived_bonds"])


@pytest.mark.cuda
def test_device_perception_matches_cpu_on_stock_corpus() -> None:
    """Bit-exact GPU/CPU perception across the PaRoutes n1 stock corpus: atom
    aromatic masks and aromatic bond bits must agree on every molecule. This is
    the device parity gate for the aromaticity port."""
    _skip_if_no_cuda()

    from tests.datadirs import external_data_root

    stock_path = (
        external_data_root() / "external" / "paroutes" / "processed" / "n1-stock.smi.gz"
    )

    seeds = []
    smiles_list = []
    with gzip.open(stock_path, "rt") as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            smiles = fields[0]
            try:
                seed = _perception_seed(smiles)
            except Exception:
                continue
            if int(seed["z"].shape[0]) > _BUCKET:
                continue
            seeds.append(seed)
            smiles_list.append(smiles)
            if len(seeds) >= 1000:
                break

    assert len(seeds) >= 900
    out_a, out_bonds, status = _run_perceive_batch(seeds)

    for idx, seed in enumerate(seeds):
        n = int(seed["z"].shape[0])
        assert int(status[idx]) == 0, smiles_list[idx]
        assert np.array_equal(out_a[idx, :n], seed["perceived_a"]), smiles_list[idx]
        np.testing.assert_array_equal(
            out_bonds[idx, :n, :n], seed["perceived_bonds"], err_msg=smiles_list[idx]
        )
