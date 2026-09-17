"""Task D2 tests: NATIVE CDPKit tautomer + protonation prep in CDPKitConformerGenerator.

Run: ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests_prep_cdpkit
"""
import os
import tempfile
import unittest

from rdkit import Chem  # import FIRST (before CDPL) for GLIBCXX safety

try:
    import CDPL.Chem  # noqa: F401

    _CDPKIT = True
except ImportError:
    _CDPKIT = False

from drugex.training.scorers.conformer_generators import CDPKitConformerGenerator


def _first_sdf_charge(path):
    mols = [m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
    return Chem.GetFormalCharge(mols[0]) if mols else None


@unittest.skipUnless(_CDPKIT, "CDPKit not available")
class CDPKitConformerPrep(unittest.TestCase):
    """genConformers should tautomer-canonicalize + protonate (pH 7.4) via NATIVE CDPKit."""

    def test_protonate_on_yields_anion(self):
        cg = CDPKitConformerGenerator(max_conformers=2, max_isomers=1, protonate=True)
        with tempfile.TemporaryDirectory() as d:
            sdf = cg.genConformers(["CC(=O)O"], d)  # acetic acid
            self.assertEqual(_first_sdf_charge(sdf), -1)  # -> carboxylate

    def test_protonate_off_stays_neutral(self):
        cg = CDPKitConformerGenerator(max_conformers=2, max_isomers=1, protonate=False)
        with tempfile.TemporaryDirectory() as d:
            sdf = cg.genConformers(["CC(=O)O"], d)
            self.assertEqual(_first_sdf_charge(sdf), 0)

    def test_base_becomes_cation(self):
        cg = CDPKitConformerGenerator(max_conformers=2, max_isomers=1, protonate=True)
        with tempfile.TemporaryDirectory() as d:
            sdf = cg.genConformers(["CCN"], d)  # ethylamine
            self.assertEqual(_first_sdf_charge(sdf), +1)  # -> ammonium


if __name__ == "__main__":
    unittest.main(verbosity=2)
