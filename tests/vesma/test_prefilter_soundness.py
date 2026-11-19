"""The folded-path prefilter must never prune a pair the dense matcher would match.

The prefilter decides which (pattern, molecule) pairs the matcher is even asked about, so an
unsound one loses chemistry *silently*: no refusal, no counter, no divergence anywhere -- the
route simply never appears. That is the worst failure shape in the engine, and it is the reason
the soundness claim gets a test over the rules that actually ship rather than an argument.

Why the claim holds today, so that a future change can be checked against it. A path feature is
(atom labels along the path, bond classes between them). Both sides reduce every present bond to
one class -- `invariants.bond_feature_classes` returns `(FLEX_BOND,)` whatever its `query`
argument says, and both device ports do the same -- so bond *order* is not in the feature at all,
and a query bond can never fail to find its counterpart. Atom labels come from
`exact_single_z_from_atom`, which answers -1 for any atom that is not one concrete element, and a
path touching a -1 emits nothing. So a query atom like `[C,N]` *removes* features from the query
fingerprint rather than adding a wrong one. Both halves make the query's fingerprint a subset of
what the molecule's would contain, which is exactly subset-soundness.

Both halves are also why the prefilter is weaker than its machinery suggests: the
`SINGLE_/DOUBLE_/TRIPLE_BOND_CLASS` constants in both shaders are dead, and the `options[7][4]`
variant expansion expands nothing. Making the query side emit real bond variants would make the
prefilter sharper -- and getting it wrong would prune a true candidate. This test is what makes
attempting that safe, which is the only reason to write it before anyone does.

`tests/vesma/test_metal_prefilter.py` makes the same comparison over the generated
retro-benchmark corpus. This one uses the shipped library, because that is where the wildcard and
multi-element query atoms the argument rests on actually live.
"""

from __future__ import annotations

import csv

import pytest

from tests.datasets import external_data_root
from vesma import cmol_from_smiles, gmol_from_cmol, gmol_from_smarts

# No module-level `slow`: the two host tests are a second over the whole shipped library and
# belong in the default run, which is where a prefilter change would be made. Only the device
# case carries `gpu`/`metal`.

TEMPLATES = external_data_root() / "templates"
RULE_SETS = (
    "templates-hartenfeller",
    "templates-medchem",
    "templates-protecting-groups",
)

#: Targets chosen to carry the functionality the shipped rules disconnect: an amide and an ester,
#: a sulfonamide, a biaryl, a Boc-protected amine, a halopyridine, a nitrile and a macrocycle.
TARGETS = (
    "CN(C(=O)C(C)(C)c1cc(Cl)cc(Cl)c1)c1cnc(N2CCC[C@H]2CO)cc1-c1ccccc1Cl",
    "COC(=O)c1cccc(C(C)(C)C#N)c1",
    "CC(C)(C)OC(=O)N1CCc2cc(Br)ccc21",
    "O=S(=O)(Nc1ccc(F)cc1)c1ccc(C)cc1",
    "O=C(CCCNC1=O)NCCOC2=NN3C(C=C2C4=CN(C)N=C4)=NC=C3C5=CC1=CN=C5",
    "OCc1ccccc1I",
)


def _lhs_queries() -> list[str]:
    """Every shipped rule's left-hand side, which is what the prefilter screens."""
    seen: list[str] = []
    for name in RULE_SETS:
        with (TEMPLATES / f"{name}.csv").open() as handle:
            for row in csv.DictReader(handle):
                smarts = row["retro_template"].split(">>")[0].strip()
                if smarts:
                    seen.append(smarts)
    return seen


def test_the_shipped_library_has_the_query_atoms_this_rests_on() -> None:
    """Guard the guard: a corpus of only concrete atoms would prove nothing.

    The soundness argument turns on query atoms whose label is -1, so the test is only
    meaningful if the shipped rules contain some. If a future library lost them, this test
    would pass for the wrong reason and should be told so.
    """
    queries = _lhs_queries()
    assert len(queries) > 300, len(queries)
    unlabelable = [q for q in queries if "*" in q or "," in q or "#" in q]
    assert len(unlabelable) > 20, (
        "no multi-element or wildcard query atoms in the shipped library, so this file's "
        f"soundness claim is untested by it; found {len(unlabelable)}"
    )


@pytest.mark.gpu
@pytest.mark.metal
@pytest.mark.parametrize("target", TARGETS)
def test_the_metal_prefilter_prunes_no_pair_the_dense_matcher_matches(
    target: str,
) -> None:
    from vesma.metal_matcher import (
        GpuUnavailableError,
        MetalFoldedPrefilteredMatcher,
        MetalMatcher,
        probe_metal,
    )

    try:
        probe_metal()
    except (GpuUnavailableError, RuntimeError) as exc:
        pytest.skip(f"Metal backend not available: {exc}")

    molecule = gmol_from_cmol(cmol_from_smiles(target)[0])
    patterns = [gmol_from_smarts(smarts)[0] for smarts in _lhs_queries()]

    dense = MetalMatcher().has_match_many(patterns, [molecule])
    screened = MetalFoldedPrefilteredMatcher().has_match_many(patterns, [molecule])

    assert dense.shape == screened.shape
    lost = [
        _lhs_queries()[i]
        for i in range(dense.shape[0])
        if dense[i, 0] and not screened[i, 0]
    ]
    assert not lost, f"prefilter pruned {len(lost)} true match(es): {lost[:5]}"
    # And the test has to be able to fail: a target nothing matches proves nothing.
    assert dense.sum() > 0, target


def test_the_cpu_prefilter_prunes_no_pair_the_dense_matcher_matches() -> None:
    """The same claim on the host, where no device is needed to check it.

    The device ports are faithful to `invariants.bond_feature_classes`, so this is where the
    claim actually lives; the Metal case above proves the port did not weaken it.
    """
    from vesma.cpu_matcher import find_matches
    from vesma.pathfp import FoldedFingerprintConfig, folded_path_fingerprint

    config = FoldedFingerprintConfig()
    queries = _lhs_queries()
    compiled = [gmol_from_smarts(smarts)[0] for smarts in queries]

    checked = 0
    for target in TARGETS:
        molecule = gmol_from_cmol(cmol_from_smiles(target)[0])
        molecule_fp = folded_path_fingerprint(molecule, query=False, config=config)
        for smarts, pattern in zip(queries, compiled, strict=True):
            if find_matches(pattern, molecule, return_first=True).status != "match":
                continue
            checked += 1
            query_fp = folded_path_fingerprint(pattern, query=True, config=config)
            assert not (query_fp & ~molecule_fp).any(), (
                f"{smarts!r} matches {target!r} but its fingerprint has bits the "
                "molecule's does not, so the prefilter would prune a true match"
            )
    assert checked > 0, "no matches across the corpus, so nothing was actually checked"
