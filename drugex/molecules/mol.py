"""
core

Created by: Martin Sicho
On: 21.04.22, 22:03
"""

from rdkit import Chem

from drugex.molecules.interfaces import Molecule


class InitializationException(Exception):
    pass

class DrExMol(Molecule):

    def __init__(self, rd_mol, identifier=None):
        if not rd_mol:
            raise InitializationException(f"Attempting to create a DrExMol instance with and empty molecule! Identifier: {identifier}")

        self.rd_mol = rd_mol
        self.canonical_smiles = identifier
        if not self.canonical_smiles:
            self.canonical_smiles = Chem.MolToSmiles(rd_mol, False, False, -1, True, False, False, False)

    @property
    def smiles(self):
        return self.canonical_smiles

    def annotate(self, key, value):
        self.rd_mol.SetProp(key, value)

    def getAnnotation(self, key):
        return self.rd_mol.GetProp(key)

    def getMetadata(self):
        props = self.rd_mol.GetPropNames()
        return  {key : self.rd_mol.GetProp(key) for key in props}

    def getUniqueID(self):
        return self.canonical_smiles

    def asRDKit(self):
        return self.rd_mol


