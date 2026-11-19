"""Device chemistry measured against the CPU model it is a port of.

The CPU chemistry is frozen precisely so the GPU ports have a fixed target, so these tests
ask the device what it thinks and compare that to ``vesma`` itself rather than to a golden
file. A golden file would record what the device did on the day it was written; comparing
against the model records whether the device still agrees with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from vesma.configurations import (
    CONFIG_CHARGE_MAX,
    CONFIG_CHARGE_MIN,
    CONFIG_CHARGE_SPAN,
    CONFIG_PI_BIAS,
    CONFIG_Z_SPAN,
    Configuration,
    ConfigurationFailure,
    configuration_table,
    configurations_for,
    select_circuit_configuration,
    select_configuration,
    unwritten_hydrogen_configuration,
    unwritten_hydrogens,
)

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def skip_if_no_cuda() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")


# Every element the configuration table declares, plus states it declares nothing about --
# an undeclared element must come back as a *different* failure from a declared element
# whose states do not fit, and only exercising both proves the device keeps them apart.
DECLARED_Z = (5, 6, 7, 8, 9, 14, 15, 16, 17, 32, 33, 34, 35, 50, 51, 52, 53)
METAL_Z = (3, 11, 12, 19, 25, 29, 30, 50, 76)
UNDECLARED_Z = (0, 1, 2, 13, 26, 47, 99)
PROBE_Z = tuple(sorted(set(DECLARED_Z + METAL_Z + UNDECLARED_Z)))


def _probe_domain() -> np.ndarray:
    """Written atom states spanning the domain the packed atom words can express."""
    grids = np.meshgrid(
        np.array(PROBE_Z, dtype=np.int32),
        np.arange(-3, 5, dtype=np.int32),  # the mask_for_charge clamp domain
        np.arange(0, 6, dtype=np.int32),  # sigma degree
        np.arange(0, 7, dtype=np.int32),  # bond-order sum
        np.arange(0, 5, dtype=np.int32),  # total hydrogen
        np.arange(0, 2, dtype=np.int32),  # delocalized
        indexing="ij",
    )
    return np.stack([grid.reshape(-1) for grid in grids], axis=1).astype(np.int32)


def _expected_status(outcome: Configuration | ConfigurationFailure) -> int:
    """The device status word: zero for a selection, otherwise the failure, shifted."""
    if isinstance(outcome, ConfigurationFailure):
        return int(outcome) + 1
    return 0


def _expected_row(
    outcome: Configuration | ConfigurationFailure,
) -> tuple[int, int, int]:
    if isinstance(outcome, ConfigurationFailure):
        return (0, 0, 0)
    return (
        int(outcome.sigma_capacity),
        int(outcome.graph_valence),
        int(outcome.pi) + 2,
    )


def test_device_configuration_table_matches_the_model() -> None:
    """The packed table reproduces ``configurations_for`` entry for entry.

    Packing is where a table stops being the thing it was generated from, so this checks
    the bytes rather than trusting the generator.
    """
    index, rows = configuration_table()
    for z in range(CONFIG_Z_SPAN):
        for charge in range(CONFIG_CHARGE_MIN, CONFIG_CHARGE_MAX + 1):
            slot = int(index[z * CONFIG_CHARGE_SPAN + (charge - CONFIG_CHARGE_MIN)])
            offset, count = slot & 0xFFFF, slot >> 16
            expected = configurations_for(z, charge)
            assert count == len(expected), (z, charge)
            for position, configuration in enumerate(expected):
                word = int(rows[offset + position])
                assert (
                    word & 0xFF,
                    (word >> 8) & 0xFF,
                    ((word >> 16) & 0xFF) - CONFIG_PI_BIAS,
                ) == (
                    configuration.sigma_capacity,
                    configuration.graph_valence,
                    int(configuration.pi),
                )


def test_device_selectors_agree_with_the_configuration_model() -> None:
    """All three selectors, on device, over the whole expressible domain.

    Both what was selected *and* which failure was reported: an undeclared element and a
    declared one whose states do not fit are different answers, and collapsing them would
    turn a gap in the model into a claim about the molecule.
    """
    skip_if_no_cuda()
    from vesma.cuda_retro import probe_cuda_configurations

    queries = _probe_domain()
    observed = probe_cuda_configurations(queries)

    for query, row in zip(queries.tolist(), observed.tolist(), strict=True):
        z, charge, sigma_degree, bond_order_sum, total_h, delocalized = query
        context = (
            f"z={z} charge={charge} d={sigma_degree} b={bond_order_sum} h={total_h}"
        )

        selected = select_configuration(
            z=z,
            charge=charge,
            sigma_degree=sigma_degree,
            bond_order_sum=bond_order_sum,
            total_h=total_h,
            delocalized=bool(delocalized),
        )
        expected = _expected_row(selected)
        assert row[0] == _expected_status(selected), f"select_configuration {context}"
        assert tuple(row[1:4]) == expected, f"select_configuration {context}"

        circuit = select_circuit_configuration(
            z=z, charge=charge, sigma_degree=sigma_degree, total_h=total_h
        )
        expected = _expected_row(circuit)
        assert row[4] == _expected_status(circuit), f"select_circuit {context}"
        assert tuple(row[5:8]) == expected, f"select_circuit {context}"

        unwritten = unwritten_hydrogen_configuration(
            z=z,
            charge=charge,
            sigma_degree=sigma_degree,
            bond_order_sum=bond_order_sum,
            delocalized=bool(delocalized),
        )
        expected = _expected_row(unwritten)
        assert row[8] == _expected_status(unwritten), f"unwritten {context}"
        assert tuple(row[9:12]) == expected, f"unwritten {context}"
        if not isinstance(unwritten, ConfigurationFailure):
            hydrogens = unwritten_hydrogens(
                unwritten,
                sigma_degree=sigma_degree,
                bond_order_sum=bond_order_sum,
                delocalized=bool(delocalized),
            )
            assert row[12] == hydrogens, f"unwritten hydrogens {context}"


def test_the_isoelectronic_shift_survives_the_bake() -> None:
    """An O+ does not inherit nitrogen's octet expansion.

    The shift governs valence (looked up at ``z - charge``) but not expansion eligibility
    or outer electrons (the atom's own, at ``z``). A device that re-derived the rule could
    apply it uniformly and quietly gain a hypervalent oxygen; one that reads the baked
    answer cannot. Worth pinning explicitly because the two halves look symmetric.
    """
    skip_if_no_cuda()
    from vesma.cuda_retro import probe_cuda_configurations

    # O+ shares nitrogen's electron count, so it reaches valence 3 -- but never nitrogen's
    # absent expansion, and never phosphorus's valence 5.
    assert {c.graph_valence for c in configurations_for(8, 1)} == {3}
    # S+ shares phosphorus's count and *does* reach phosphorus's expanded valence 5, which
    # is the sulfoxonium of the Corey-Chaykovsky reagent.
    assert 5 in {c.graph_valence for c in configurations_for(16, 1)}

    queries = np.array(
        [
            [8, 1, 3, 5, 0, 0],  # a hypervalent O+ that must not exist
            [16, 1, 3, 5, 0, 0],  # the sulfoxonium S+ that must
        ],
        dtype=np.int32,
    )
    observed = probe_cuda_configurations(queries)
    assert observed[0][0] != 0, "device accepted a valence-5 oxygen cation"
    assert observed[1][0] == 0, "device refused the sulfoxonium sulfur"
