"""Task D3 tests: NATIVE OpenEye (OEQuacPac) tautomer + protonation in OmegaConformerGenerator.

Run: ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests_prep_openeye
"""
import os
import tempfile
import unittest

from rdkit import Chem  # noqa: F401  (import order safety)

try:
    from openeye import oechem, oequacpac

    _OE = oechem.OEChemIsLicensed() and oequacpac.OEQuacPacIsLicensed()
except Exception:
    _OE = False

from drugex.training.scorers.conformer_generators import OmegaConformerGenerator


def _first_oeb_charge(path):
    from openeye import oechem as oe

    ifs = oe.oemolistream()
    if not ifs.open(path):
        return None
    mol = oe.OEMol()
    if not oe.OEReadMolecule(ifs, mol):
        return None
    return oe.OENetCharge(mol)


@unittest.skipUnless(_OE, "OpenEye toolkit / OEQuacPac not licensed")
class OmegaConformerPrep(unittest.TestCase):
    """genConformers should tautomer + protonate (pH 7.4) via NATIVE OEQuacPac before Omega."""

    def test_protonate_on_yields_anion(self):
        cg = OmegaConformerGenerator(max_conformers=2, max_centers=1, protonate=True)
        with tempfile.TemporaryDirectory() as d:
            out = cg.genConformers(["CC(=O)O"], d)  # acetic acid
            self.assertEqual(_first_oeb_charge(out), -1)

    def test_protonate_off_stays_neutral(self):
        cg = OmegaConformerGenerator(max_conformers=2, max_centers=1, protonate=False,
                                     canonicalize_tautomer=False)
        with tempfile.TemporaryDirectory() as d:
            out = cg.genConformers(["CC(=O)O"], d)
            self.assertEqual(_first_oeb_charge(out), 0)

    def test_base_becomes_cation(self):
        cg = OmegaConformerGenerator(max_conformers=2, max_centers=1, protonate=True)
        with tempfile.TemporaryDirectory() as d:
            out = cg.genConformers(["CCN"], d)  # ethylamine
            self.assertEqual(_first_oeb_charge(out), +1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
