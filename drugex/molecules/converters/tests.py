"""
tests.py

Created by: Martin Sicho
On: 22.04.22, 13:40
"""
from unittest import TestCase

from drugex.molecules.converters.standardizers import DrExStandardizer
from drugex.molecules.mol import DrExMol
from drugex.molecules.suppliers import StandardizedSupplier, TestSupplier


class TestStandardizer(TestCase):

    def test_default(self):
        mols = TestSupplier(["c1ccccc1CN(=O)", "CC(=O)[O-]", "CCCCn1cc[n+](c1)C.F[P-](F)(F)(F)(F)F"])

        standardized_mols = StandardizedSupplier(mols, standardizer=DrExStandardizer())

        expected = (
            "O=NCc1ccccc1", # canonical
            "CC(=O)O", # without charge
            "CCCCn1cc[n+](C)c1", # remove salt
            # TODO: add more cases
        )
        for mol, expected_mol in zip(standardized_mols, expected):
            self.assertTrue(mol == expected_mol)

    def test_with_converter(self):
        mols = TestSupplier(["c1ccccc1CN(=O)", "CC(=O)[O-]", "CCCCn1cc[n+](c1)C.F[P-](F)(F)(F)(F)F"])
        standardized_mols = StandardizedSupplier(
            mols,
            standardizer=DrExStandardizer(output='DrExMol')
        )
        for mol in standardized_mols:
            self.assertTrue(isinstance(mol, DrExMol))