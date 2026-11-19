import numpy as np

from vesma import cmol_from_smiles, reaction_patterns_from_smarts, smiles_from_cmol
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.matcher import ReactionMatcher


def test_readme_reaction_application() -> None:
    product_smiles = [
        "CC(=O)OC",
        "COC(=O)CC(=O)OC",
    ]
    products = [cmol_from_smiles(smiles)[0] for smiles in product_smiles]
    reaction_smarts = ["[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])-[O:3].[C:4]"]
    rules = reaction_patterns_from_smarts(reaction_smarts)
    matcher: ReactionMatcher = CpuReactionMatcher(rules)

    precursors, hits = matcher.match(products)

    assert [smiles_from_cmol(molecule) for molecule in precursors] == [
        # Bracketed atoms state their hydrogens, as OpenSMILES 3.6 requires: the acid
        # is `[OH:3]` and the methane `[CH4:4]`. Written `[O:3]`/`[C:4]` these would read
        # back as an acyl radical and atomic carbon.
        "C-[C:1](-[OH:3])=[O:2]",
        "[CH4:4]",
        "[CH4:4]",
        "[OH:3]-[C:1](-C-C(-O-C)=O)=[O:2]",
        "C-O-C(-C-[C:1](-[OH:3])=[O:2])=O",
        "[CH4:4]",
    ]
    # Co-precursors of one application share the first three columns; the fourth points
    # at this backend's dense precursor batch, one row per hit.
    np.testing.assert_array_equal(
        hits,
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 2],
            [1, 0, 0, 3],
            [1, 0, 1, 4],
            [1, 0, 1, 5],
        ],
    )
