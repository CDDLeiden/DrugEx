"""
tests.py

Created by: Martin Sicho
On: 22.04.22, 13:40
"""
from unittest import TestCase

from drugex.molecules.converters.standardizers import DefaultStandardizer
from drugex.molecules.mol import DrExMol
from drugex.molecules.suppliers import StandardizedSupplier, ListSupplier


class TestStandardizer(TestCase):

    def test_default(self):
        """
        Tests whether the default standardizer is working as expected.

        Returns: None
        """

        standardized_mols = StandardizedSupplier(["c1ccccc1CN(=O)", "CC(=O)[O-]", "CCCCn1cc[n+](c1)C.F[P-](F)(F)(F)(F)F"], standardizer=DefaultStandardizer())

        expected = (
            "O=NCc1ccccc1", # canonical
            "CC(=O)O", # without charge
            "CCCCn1cc[n+](C)c1", # remove salt
            # TODO: add more cases
        )
        for mol, expected_mol in zip(standardized_mols, expected):
            self.assertTrue(mol == expected_mol)