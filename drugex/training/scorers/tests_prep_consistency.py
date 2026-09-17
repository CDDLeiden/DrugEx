"""Task D / #11: cross-backend protonation CONSISTENCY.

Confirms the three NATIVE protonators (RDKit=Dimorphite-DL, CDPKit=ProtonationStateStandardizer,
OpenEye=OEQuacPac) agree on the dominant pH-7.4 net charge for the same molecule — the basis
for the "native-per-backend is consistent" claim.

Run: ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests_prep_consistency
"""
import tempfile
import unittest

from rdkit import Chem  # FIRST

try:
    import dimorphite_dl  # noqa: F401
    _DIMORPHITE = True
except ImportError:
    _DIMORPHITE = False
try:
    import CDPL.Chem  # noqa: F401
    _CDPKIT = True
except ImportError:
    _CDPKIT = False
try:
    from openeye import oechem, oequacpac
    _OE = oechem.OEChemIsLicensed() and oequacpac.OEQuacPacIsLicensed()
except Exception:
    _OE = False

from drugex.training.scorers.conformer_generators import (
    RDKitConformerGenerator, CDPKitConformerGenerator, OmegaConformerGenerator)

# (SMILES, expected dominant net charge at pH 7.4)
CASES = [("CC(=O)O", -1), ("CCN", +1), ("c1ccccc1", 0)]


def _sdf_charge(path):
    mols = [m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
    return Chem.GetFormalCharge(mols[0]) if mols else None


def _oeb_charge(path):
    from openeye import oechem as oe
    ifs = oe.oemolistream()
    if not ifs.open(path):
        return None
    mol = oe.OEMol()
    return oe.OENetCharge(mol) if oe.OEReadMolecule(ifs, mol) else None


class CrossBackendProtonationConsistency(unittest.TestCase):
    def _charge(self, backend, smi):
        with tempfile.TemporaryDirectory() as d:
            if backend == "RDKit":
                cg = RDKitConformerGenerator(max_conformers=1, max_isomers=1, protonate=True)
                return _sdf_charge(cg.genConformers([smi], d))
            if backend == "CDPKit":
                cg = CDPKitConformerGenerator(max_conformers=1, max_isomers=1, protonate=True)
                return _sdf_charge(cg.genConformers([smi], d))
            cg = OmegaConformerGenerator(max_conformers=1, max_centers=1, protonate=True)
            return _oeb_charge(cg.genConformers([smi], d))

    def test_all_available_backends_agree_on_dominant_charge(self):
        backends = (["RDKit"] if _DIMORPHITE else []) + (["CDPKit"] if _CDPKIT else []) + (["OpenEye"] if _OE else [])
        self.assertGreaterEqual(len(backends), 2, "need >=2 backends to compare")
        for smi, expected in CASES:
            charges = {b: self._charge(b, smi) for b in backends}
            for b, q in charges.items():
                self.assertEqual(q, expected, f"{b} gave {q} for {smi}, expected {expected}; all={charges}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
