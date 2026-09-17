"""Task D1 tests: native tautomer + protonation prep in RDKitConformerGenerator.

Run: ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests_prep_rdkit
"""
import os
import tempfile
import unittest

from rdkit import Chem

try:
    import dimorphite_dl  # noqa: F401

    _DIMORPHITE = True
except ImportError:
    _DIMORPHITE = False

from drugex.training.scorers.conformer_generators import RDKitConformerGenerator


def _first_sdf_mol(path):
    mols = [m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
    return mols[0] if mols else None


class RDKitConformerPrep(unittest.TestCase):
    """genConformers should tautomer-canonicalize then protonate (pH 7.4) before embedding."""

    @unittest.skipUnless(_DIMORPHITE, "dimorphite_dl not available")
    def test_protonate_on_yields_anion_conformer(self):
        with tempfile.TemporaryDirectory() as d:
            cg = RDKitConformerGenerator(max_conformers=2, max_isomers=1, protonate=True)
            sdf = cg.genConformers(["CC(=O)O"], d)  # acetic acid
            mol = _first_sdf_mol(sdf)
            self.assertIsNotNone(mol)
            self.assertEqual(Chem.GetFormalCharge(mol), -1)  # -> carboxylate

    def test_protonate_off_stays_neutral(self):
        with tempfile.TemporaryDirectory() as d:
            cg = RDKitConformerGenerator(max_conformers=2, max_isomers=1, protonate=False)
            sdf = cg.genConformers(["CC(=O)O"], d)
            mol = _first_sdf_mol(sdf)
            self.assertIsNotNone(mol)
            self.assertEqual(Chem.GetFormalCharge(mol), 0)

    def test_tautomer_canonicalized(self):
        # enol of 2-butanone -> canonical keto tautomer CCC(C)=O
        with tempfile.TemporaryDirectory() as d:
            cg = RDKitConformerGenerator(
                max_conformers=2, max_isomers=1, canonicalize_tautomer=True, protonate=False
            )
            sdf = cg.genConformers(["CC(O)=CC"], d)
            mol = _first_sdf_mol(sdf)
            self.assertIsNotNone(mol)
            got = Chem.CanonSmiles(Chem.MolToSmiles(Chem.RemoveHs(mol)))
            self.assertEqual(got, Chem.CanonSmiles("CCC(C)=O"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
