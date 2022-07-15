"""
gcmol

Created by: Martin Sicho
On: 10.06.22, 16:50
"""

#THESE FUNCTIONS WERE ADDED BY HELLE FROM:
#   https://github.com/BenevolentAI/guacamol/blob/8247bbd5e927fbc3d328865d12cf83cb7019e2d6/guacamol/utils/data.py#L11
# to solve AttributeError: module 'utils' has no attribute 'canonicalize_list'
from typing import Optional, Iterable, List

from rdkit import Chem


def canonicalize(smiles: str, include_stereocenters=True) -> Optional[str]:
    """
    Canonicalize the SMILES strings with RDKit.
    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543
    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string
    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None

def remove_duplicates(list_with_duplicates):
    """
    Removes the duplicates and keeps the ordering of the original list.
    For duplicates, the first occurrence is kept and the later occurrences are ignored.
    Args:
        list_with_duplicates: list that possibly contains duplicates
    Returns:
        A list with no duplicates.
    """

    unique_set = set()
    unique_list = []
    for element in list_with_duplicates:
        if element not in unique_set:
            unique_set.add(element)
            unique_list.append(element)

    return unique_list

def canonicalize_list(smiles_list: Iterable[str], include_stereocenters=True) -> List[str]:
    """
    Canonicalize a list of smiles. Filters out repetitions and removes corrupted molecules.
    Args:
        smiles_list: molecules as SMILES strings
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES strings
    Returns:
        The canonicalized and filtered input smiles.
    """

    canonicalized_smiles = [canonicalize(smiles, include_stereocenters) for smiles in smiles_list]

    # Remove None elements
    canonicalized_smiles = [s for s in canonicalized_smiles if s is not None]

    return remove_duplicates(canonicalized_smiles)
