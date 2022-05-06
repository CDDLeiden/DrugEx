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
        self.mols = supplier if hasattr(supplier, "__next__") else iter(supplier)

    def next(self):
        return next(self.mols)

class DataFrameSupplier(BaseMolSupplier):

     def __init__(
             self,
             df,
             mol_col,
             extra_cols = tuple(),
             converter=SmilesToDrEx(),
             hide_duplicates=False
     ):
        super().__init__(converter=converter, hide_duplicates=hide_duplicates)
        # df.drop(df.columns.difference(extra_cols + (mol_col,)), 1, inplace=True)
        self.mols = df.iterrows()
        self.mol_col = mol_col
        self.extra_cols = extra_cols

     def next(self):
         row = next(self.mols)[1]
         mol = getattr(row, self.mol_col)
         mol_data = {key : getattr(row, key) for key in self.extra_cols}
         return mol, mol_data

     def __getstate__(self):
        d = self.__dict__.copy()
        if 'mols' in d:
            d['mols'] = repr(d['mols'])
        return d

     def __setstate__(self, d):
        if 'mols' in d:
            d['mols'] = None
        self.__dict__.update(d)

class  TestSupplier(BaseMolSupplier):

    def __init__(self, smiles):
        super().__init__(converter=SmilesToDrEx(), hide_duplicates=False)
        self.mols = iter(smiles)


    def next(self):
        return next(self.mols)