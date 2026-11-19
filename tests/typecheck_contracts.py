from typing import assert_type

from rdkit.Chem.rdchem import Mol as RdMol

from vertsy.core import Step
from vertsy.expansions import Expansions, Steps, expansions_of, materialize
from vesma.cmol import CMol
from vesma.keyer import (
    Keyer,
    aromatic_wl_keyer,
    cmol_rdkit_smiles_keyer,
    cmol_smiles_keyer,
    rdkit_inchi_keyer,
    rdkit_inchikey_keyer,
    rdkit_smiles_keyer,
    short_ring_wl_keyer,
)
from vesma.wl import WlToken

assert_type(short_ring_wl_keyer, Keyer[CMol, WlToken])
assert_type(aromatic_wl_keyer, Keyer[CMol, WlToken])
assert_type(cmol_smiles_keyer, Keyer[CMol, str])
assert_type(cmol_rdkit_smiles_keyer, Keyer[CMol, str])
assert_type(rdkit_smiles_keyer, Keyer[RdMol, str])
assert_type(rdkit_inchi_keyer, Keyer[RdMol, str])
assert_type(rdkit_inchikey_keyer, Keyer[RdMol, str])

step: Step[str] = (0, frozenset({"precursor"}))
expansions = expansions_of({"product": (step,)})
assert_type(expansions, Expansions[str])
assert_type(materialize(expansions), dict[str, Steps[str]])
