"""
suppliers

Created by: Martin Sicho
On: 22.04.22, 13:50
"""
from drugex.molecules.converters.default import SmilesToDrEx
from drugex.molecules.interfaces import BaseMolSupplier


class StandardizedSupplier(BaseMolSupplier):

    def __init__(self, supplier, standardizer):
        super().__init__(converter=standardizer)
        self.mols = supplier

    def next(self):
        return next(self.mols)


class  TestSupplier(BaseMolSupplier):

    def __init__(self, smiles):
        super().__init__(converter=SmilesToDrEx(), hide_duplicates=False)
        self.mols = iter(smiles)


    def next(self):
        return next(self.mols)