"""Tests for the shared pH-protonation primitive (Task D).

Run: ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests_protonation
"""
import unittest

from rdkit import Chem  # import FIRST (before dimorphite/CDPL) for GLIBCXX safety

try:
    import dimorphite_dl  # noqa: F401

    _DIMORPHITE = True
except ImportError:
    _DIMORPHITE = False

from drugex.training.scorers.protonation import protonate_smiles


def _charge(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.GetFormalCharge(m) if m is not None else None


@unittest.skipUnless(_DIMORPHITE, "dimorphite_dl not available")
class ProtonateSmiles(unittest.TestCase):
    """pH-7.4 protonation: acids -> anion, bases -> cation, neutral unchanged."""

    def test_carboxylic_acid_becomes_anion(self):
        out = protonate_smiles("CC(=O)O", ph=7.4)  # acetic acid, pKa ~4.8
        self.assertIsInstance(out, str)
        self.assertEqual(_charge(out), -1)

    def test_aliphatic_amine_becomes_cation(self):
        out = protonate_smiles("CCN", ph=7.4)  # ethylamine, pKa ~10.6
        self.assertEqual(_charge(out), +1)

    def test_neutral_molecule_unchanged(self):
        out = protonate_smiles("c1ccccc1", ph=7.4)  # benzene, no ionizable group
        self.assertEqual(_charge(out), 0)

    def test_list_input_returns_list_same_length(self):
        out = protonate_smiles(["CC(=O)O", "c1ccccc1", "CCN"], ph=7.4)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 3)
        self.assertEqual(_charge(out[0]), -1)  # acid -> anion
        self.assertEqual(_charge(out[1]), 0)  # benzene -> neutral
        self.assertEqual(_charge(out[2]), +1)  # amine -> cation

    def test_invalid_smiles_falls_back_gracefully(self):
        # garbage in -> returned unchanged, no exception (pipeline robustness)
        out = protonate_smiles("not_a_smiles", ph=7.4)
        self.assertEqual(out, "not_a_smiles")


if __name__ == "__main__":
    unittest.main(verbosity=2)
