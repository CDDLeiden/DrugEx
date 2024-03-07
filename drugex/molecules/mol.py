from rdkit import Chem

from drugex.molecules.interfaces import Molecule


class InitializationException(Exception):
    pass

class DrExMol(Molecule):

    def __init__(self, rd_mol, identifier=None):
        if not rd_mol:
            raise InitializationException(f"Attempting to create a DrExMol instance with an empty molecule! Identifier was: {identifier}")

        self.rd_mol = rd_mol
        self.canonical_smiles = identifier
        if not self.canonical_smiles:
            self.canonical_smiles = Chem.MolToSmiles(rd_mol, False, False, -1, True, False, False, False)
        self.annotations = dict()

    @property
    def smiles(self):
        return self.canonical_smiles

    def annotate(self, key, value):
        self.annotations[key] = value

    def getAnnotation(self, key):
        return self.annotations[key]

    def getMetadata(self):
        return self.annotations

    def getUniqueID(self):
        return self.canonical_smiles

    def asRDKit(self):
        return self.rd_mol


