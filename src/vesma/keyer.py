from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Generic

import numpy as np

from .canonicalization import canonicalish_atom_order, reorder_cmol
from .cmol import CMol
from .cmol_builder import cmol_from_smiles
from .identity import (
    WlToken,
    aromatic_wl_molecule_id,
    aromatic_wl_token_from_smiles,
    short_ring_wl_molecule_id,
    short_ring_wl_token_from_smiles,
)
from .interop import mol_from_cmol
from .matcher import K, M
from .smiles import smiles_from_cmol
from .wl import WL_DEFAULT_ROUNDS

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol as RdMol
else:
    RdMol = Any

__all__ = [
    "Keyer",
    "aromatic_wl_keyer",
    "cmol_canonicalish_smiles",
    "cmol_rdkit_smiles_keyer",
    "cmol_smiles_keyer",
    "cmol_smiles_token_from_smiles",
    "inchikey_token_from_smiles",
    "inchi_token_from_smiles",
    "rdkit_inchikey_keyer",
    "rdkit_inchi_keyer",
    "rdkit_canonical_smiles",
    "rdkit_inchi",
    "rdkit_inchi_key",
    "rdkit_mol_canonical_smiles",
    "rdkit_mol_inchi",
    "rdkit_mol_inchi_key",
    "rdkit_smiles_keyer",
    "rdkit_token_from_smiles",
    "rdkit_version",
    "short_ring_wl_keyer",
    "vesma_version",
]


@dataclass(frozen=True)
class Keyer(Generic[M, K]):
    """A molecule identity codec from representation ``M`` to key ``K``.

    ``encode`` is the one public operation (``keyer(molecule)`` is shorthand).
    ``name``/``version`` carry the codec's identity so keyed artifacts (stocks,
    expansion stores) can be cached and serialized against the exact codec used.
    ``from_smiles`` ensures stocks and generated molecules share one identity policy.
    """

    name: str
    encode: Callable[[M], K]
    from_smiles: Callable[[str], K]
    version: Callable[[], str]

    def __call__(self, molecule: M) -> K:
        return self.encode(molecule)


def rdkit_version() -> str:
    try:
        import rdkit

        return str(rdkit.__version__)
    except ImportError:
        return "unknown"


def vesma_version() -> str:
    try:
        from importlib.metadata import version

        return version("vesma")
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# rdkit / inchi leaf encoders (both the CMol and from-SMILES sides share the
# same normalization: drop atom maps + stereo before canonicalizing).
# --------------------------------------------------------------------------- #


def normalized_rdkit_mol(rd_mol: RdMol) -> RdMol:
    from rdkit import Chem

    for atom in rd_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(rd_mol)
    return rd_mol


def rdkit_identity_mol(molecule: CMol) -> RdMol:
    from rdkit import Chem

    return normalized_rdkit_mol(Chem.Mol(mol_from_cmol(molecule)))


def rdkit_mol_identity(molecule: RdMol) -> RdMol:
    from rdkit import Chem

    return normalized_rdkit_mol(Chem.Mol(molecule))


def rdkit_mol_from_smiles(smiles: str) -> RdMol:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return normalized_rdkit_mol(mol)


def rdkit_mol_canonical_smiles(molecule: RdMol) -> str:
    return canonical_smiles_from_rdkit_mol(rdkit_mol_identity(molecule))


def canonical_smiles_from_rdkit_mol(molecule: RdMol) -> str:
    from rdkit import Chem

    # Stereo is already removed by normalized_rdkit_mol, so isomericSmiles here only
    # controls the isotope label. It stays off to agree with WL identity, which treats
    # isotopologues as one molecule; a key space that split them would disagree with
    # the identity that groups them.
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )


def rdkit_canonical_smiles(molecule: CMol) -> str:
    return canonical_smiles_from_rdkit_mol(rdkit_identity_mol(molecule))


def rdkit_token_from_smiles(smiles: str) -> str:
    return canonical_smiles_from_rdkit_mol(rdkit_mol_from_smiles(smiles))


def rdkit_mol_inchi(molecule: RdMol) -> str:
    return inchi_from_rdkit_mol(rdkit_mol_identity(molecule))


def rdkit_inchi(molecule: CMol) -> str:
    return inchi_from_rdkit_mol(rdkit_identity_mol(molecule))


def inchi_token_from_smiles(smiles: str) -> str:
    return inchi_from_rdkit_mol(rdkit_mol_from_smiles(smiles))


def inchi_from_rdkit_mol(rd_mol: RdMol) -> str:
    from rdkit.Chem import rdinchi

    inchi, return_code, message, log, _ = rdinchi.MolToInchi(rd_mol)
    if int(return_code) not in {0, 1}:
        details = str(message or log or "unknown error")
        raise ValueError(f"RDKit InChI generation failed: {details}")
    return str(inchi)


def rdkit_mol_inchi_key(molecule: RdMol) -> str:
    return inchi_key_from_rdkit_mol(rdkit_mol_identity(molecule))


def inchi_key_from_rdkit_mol(molecule: RdMol) -> str:
    from rdkit.Chem import rdinchi

    key = rdinchi.MolToInchiKey(molecule)
    if not key:
        raise ValueError("RDKit InChIKey generation failed.")
    return str(key)


def rdkit_inchi_key(molecule: CMol) -> str:
    return inchi_key_from_rdkit_mol(rdkit_identity_mol(molecule))


def inchikey_token_from_smiles(smiles: str) -> str:
    return inchi_key_from_rdkit_mol(rdkit_mol_from_smiles(smiles))


# --------------------------------------------------------------------------- #
# cmol canonical SMILES (no rdkit): canonical atom order + smiles emission.
# --------------------------------------------------------------------------- #


def cmol_canonicalish_smiles(molecule: CMol) -> str:
    """A canonical token for a molecule, written without RDKit.

    Every hydrogen is written explicitly. ``smiles_from_cmol`` brackets an atom when
    ``H - h`` is non-zero, and ``h`` records how the *input* happened to be written, not
    what the molecule is -- so ``CCCC[Sn]`` and ``CCC[CH2][Sn]`` are one molecule that
    would otherwise receive two different keys. A key that depends on spelling is the one
    thing this model exists to prevent, and the extra brackets cost only bytes in a token
    nobody reads.
    """
    order = canonicalish_atom_order(molecule)
    if not order:
        return ""
    ordered = reorder_cmol(molecule, order)
    return smiles_from_cmol(replace(ordered, h=np.zeros_like(ordered.h)))


def cmol_smiles_token_from_smiles(smiles: str) -> str:
    cmol, _ = cmol_from_smiles(smiles)
    return cmol_canonicalish_smiles(cmol)


# --------------------------------------------------------------------------- #
# WL hash keyers (one-way 128-bit tokens, built on the identity layer).
# --------------------------------------------------------------------------- #


def wl_keyer_version(codec_name: str, rounds: int) -> str:
    return f"{vesma_version()}:{codec_name}:rounds={rounds}"


short_ring_wl_keyer: Keyer[CMol, WlToken] = Keyer(
    name=f"short-ring-wl3-8-zchd-v6-r{WL_DEFAULT_ROUNDS}",
    encode=short_ring_wl_molecule_id,
    from_smiles=short_ring_wl_token_from_smiles,
    version=partial(
        wl_keyer_version,
        "short-ring-wl3-8-zchd-v6",
        WL_DEFAULT_ROUNDS,
    ),
)
aromatic_wl_keyer: Keyer[CMol, WlToken] = Keyer(
    name=f"aromatic-wl-zchd-v5-r{WL_DEFAULT_ROUNDS}",
    encode=aromatic_wl_molecule_id,
    from_smiles=aromatic_wl_token_from_smiles,
    version=partial(
        wl_keyer_version,
        "aromatic-wl-zchd-v5",
        WL_DEFAULT_ROUNDS,
    ),
)
cmol_smiles_keyer: Keyer[CMol, str] = Keyer(
    name="cmol-smiles",
    encode=cmol_canonicalish_smiles,
    from_smiles=cmol_smiles_token_from_smiles,
    version=vesma_version,
)
cmol_rdkit_smiles_keyer: Keyer[CMol, str] = Keyer(
    name="cmol-rdkit-smiles",
    encode=rdkit_canonical_smiles,
    from_smiles=rdkit_token_from_smiles,
    version=rdkit_version,
)
rdkit_smiles_keyer: Keyer[RdMol, str] = Keyer(
    name="rdkit-smiles",
    encode=rdkit_mol_canonical_smiles,
    from_smiles=rdkit_token_from_smiles,
    version=rdkit_version,
)
rdkit_inchi_keyer: Keyer[RdMol, str] = Keyer(
    name="rdkit-inchi",
    encode=rdkit_mol_inchi,
    from_smiles=inchi_token_from_smiles,
    version=rdkit_version,
)
rdkit_inchikey_keyer: Keyer[RdMol, str] = Keyer(
    name="rdkit-inchikey",
    encode=rdkit_mol_inchi_key,
    from_smiles=inchikey_token_from_smiles,
    version=rdkit_version,
)
