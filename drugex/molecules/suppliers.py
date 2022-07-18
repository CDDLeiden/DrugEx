"""
suppliers

Created by: Martin Sicho
On: 22.04.22, 13:50
"""
from drugex.molecules.converters.default import SmilesToDrEx
from drugex.molecules.interfaces import BaseMolSupplier


class StandardizedSupplier(BaseMolSupplier):
    """
    Supplies standardized molecules from the input molecules. It requires a standardizer.
    """

    def __init__(self, mols, standardizer):
        """
        Initialize a standardized supplier with the given standardizer.

        Args:
            mols: A set of input molecules. It will be converted to an iterator if not one already.
            standardizer: a `MolConverter` to transform items in 'mols' to the standardized form.
        """

        super().__init__(converter=standardizer)
        self.mols = mols if hasattr(mols, "__next__") else iter(mols)

    def next(self):
        """
        Defines access to the next item to be processed.

        Returns:
            next molecule for processing
        """

        return next(self.mols)

class  ListSupplier(BaseMolSupplier):
    """
    Basic supplier that converts molecules in a list to the desired representation (SMILES string to `DrExMol` by default).

    """

    def __init__(self, mols, converter=SmilesToDrEx()):
        """
        Initialize list supplier.

        Args:
            mols (list): A list of molecules. By default, representation as SMILES is assumed.
            converter (MolConverter): Converter to transform molecules in 'mols' to the desired representation, `SmilesToDrEx` by default.
        """
        super().__init__(converter=converter, hide_duplicates=False)
        self.mols = iter(mols)


    def next(self):
        return next(self.mols)