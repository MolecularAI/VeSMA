"""Pyinstrument call-tree profile of mol construction pipeline."""

from __future__ import annotations

import pyinstrument

from vesma import cmol_from_mol, gmol_from_cmol
from vesma.cpu_matcher import CpuMatcher
from vesma.keyer import rdkit_canonical_smiles
from vesma.reaction import (
    parse_reaction_smarts,
    precursor_signature,
    precursors_from_mapping,
)


def suppress_rdkit_errors() -> None:
    from rdkit import RDLogger

    RDLogger.logger().setLevel(RDLogger.ERROR)


RETRO_TEMPLATES = [
    "[C:1](=[O:2])-[N:3]>>[C:1](=[O:2])-O.[N:3]",
    "[c:1]-[c:2]>>[c:1]-Br.[c:2]-B(O)O",
    "[c:1]-[N:2]>>[c:1]-Br.[N:2]",
    "[C:1](=[O:2])-[O:3]-[C:4]>>[C:1](=[O:2])-O.[O:3]-[C:4]",
    "[C:1](=[O:2])-[O:3]>>[C:1](=[O:2])-Cl.[O:3]",
    "[N:1]-[C:2](=[O:3])>>[N:1].[C:2](=[O:3])-Cl",
    "[c:1]-[S:2]>>[c:1]-Br.[S:2]",
]

SAMPLE_SMILES = [
    "CC(=O)Nc1ccccc1",
    "c1ccc(-c2ccccc2)cc1",
    "c1ccc(N)cc1",
    "CC(=O)OCC",
    "O=C(O)c1ccccc1",
    "CC(=O)Nc1ccc(O)cc1",
    "c1ccc(NC(=O)c2ccccc2)cc1",
    "CC(=O)Oc1ccccc1",
    "c1ccc(-c2ccncc2)cc1",
    "c1ccc(-c2ccc(N)cc2)cc1",
    "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1",
    "CC(=O)Nc1ccc(-c2ccccc2)cc1",
    "O=C(Nc1ccccc1)c1ccc(O)cc1",
    "CCOc1ccc(NC(=O)c2ccccc2)cc1",
    "c1cc(N)c(-c2ccccc2)cc1N",
]


def main() -> None:
    suppress_rdkit_errors()
    from rdkit import Chem

    molecules = [
        cmol_from_mol(Chem.MolFromSmiles(s))
        for s in SAMPLE_SMILES
        if Chem.MolFromSmiles(s)
    ]
    templates = []
    for smarts in RETRO_TEMPLATES:
        try:
            templates.append(parse_reaction_smarts(smarts))
        except Exception:
            pass

    matcher = CpuMatcher()
    triples = []
    for mol in molecules:
        gmol = gmol_from_cmol(mol)
        for rxn in templates:
            result = matcher.find_matches(rxn.lhs, gmol, max_assignments=16)
            if result.assignments.size > 0:
                for mapping in result.assignments:
                    triples.append((mol, rxn, mapping))

    print(f"Profiling {len(triples)} triples x 500 repeats")

    profiler = pyinstrument.Profiler()
    profiler.start()

    for _ in range(500):
        for mol, rxn, mapping in triples:
            precursors = precursors_from_mapping(
                product=mol,
                reaction=rxn,
                mapping=mapping,
            )
            precursor_signature(precursors)
            for p in precursors:
                try:
                    rdkit_canonical_smiles(p)
                except Exception:
                    pass

    profiler.stop()
    print(profiler.output_text(unicode=True, color=False))


if __name__ == "__main__":
    main()
