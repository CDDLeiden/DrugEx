"""
fragments

Created by: Martin Sicho
On: 06.05.22, 15:39
"""
from drugex.molecules.interfaces import MolSupplier


class FragmentSupplier(MolSupplier):

    def __init__(self, molecules, fragmenter):
        self.molecules = molecules if hasattr(molecules, "__next__") else iter(molecules)
        self.fragmenter = fragmenter

    def next(self):
        ret = self.fragmenter(next(self.molecules))
        if ret:
            smile, frags = ret
            return {"smiles": smile, "frags" : tuple(frags)}
        else:
            return next(self)

    def convertMol(self, representation):
        return representation

    def annotateMol(self, mol, key, value):
        return mol