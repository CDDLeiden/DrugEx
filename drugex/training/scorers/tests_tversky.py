"""TDD tests for reference-weighted Tversky color/combo (Lever 4, Task #16) — CDPKit backend.

Run:  ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests_tversky

Background (HOW_CLOSE_TO_OE.md): asymmetric reference-Tversky weights the REFERENCE pharmacophore
("is the reference covered by the candidate?"), forgiving the candidate being larger than the
small CCR2 references. PheSA (Wahl 2024, 10.1021/acs.jcim.4c00516) found Tversky > Tanimoto for
enrichment. ALL THREE backends now support it:
- CDPKit: native exact (calcReference{Shape,Color}TverskyScore / calcReferenceTverskyComboScore).
- OpenEye: native Ref/Fit Tversky report columns (verified RefTverskyCombo/RefColorTversky/...).
- RDKit (#18): on rdShapeAlign's NATIVE basis via _tversky_from_tanimoto -- recovers the native
  overlap O_ab=T*(sov_a+sov_b)/(1+T) from the reported Tanimoto + native sov/sof self-overlaps
  (alpha=beta=1 round-trips to the native Tanimoto exactly). The first-order Gaussian overlap is
  an L2 inner product, so O_ab<=sqrt(sov_a*sov_b) (NOT <=min); the index is clamped to [0,1].
"""
import os
import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

try:
    import CDPL.Shape as _CDPLShape  # rdkit imported first above -> safe order
    import CDPL.Chem as _CDPLChem
    import CDPL.Pharm as _CDPLPharm
    _CDPKIT = True
except ImportError:
    _CDPKIT = False

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_REF_SDF = os.path.join(
    _REPO_ROOT, "tutorial/advanced/rocs/rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf"
)


def _cdpkit_pharm_shape(rdkit_mol):
    """Build a colored CDPKit Gaussian shape from an RDKit mol w/ a conformer."""
    import tempfile

    sdf = tempfile.mktemp(suffix=".sdf")
    w = Chem.SDWriter(sdf); w.write(rdkit_mol); w.close()
    reader = _CDPLChem.FileSDFMoleculeReader(sdf)
    cm = _CDPLChem.BasicMolecule()
    reader.read(cm)
    os.unlink(sdf)
    _CDPLPharm.prepareForPharmacophoreGeneration(cm)
    gen = _CDPLShape.GaussianShapeGenerator()
    gen.generatePharmacophoreShape(True)
    return gen.generate(cm).getElement(0)


@unittest.skipUnless(_CDPKIT, "CDPKit not available")
@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class CDPKitTverskyVariant(unittest.TestCase):
    """Task #16 — _align_and_score_helper must accept score_variant and use CDPKit's
    native Reference/Aligned Tversky scorers (exact, no approximation)."""

    @classmethod
    def setUpClass(cls):
        ref0 = [m for m in Chem.SDMolSupplier(_REF_SDF, removeHs=False) if m][0]
        cls.rshape = _cdpkit_pharm_shape(ref0)
        # triamine_acid: size-mismatched vs ref0 -> Tversky must differ from Tanimoto
        m = Chem.AddHs(Chem.MolFromSmiles("NCCNCCNc1ccc(C(=O)O)cc1"))
        p = AllChem.ETKDGv3(); p.randomSeed = 7
        AllChem.EmbedMolecule(m, p); AllChem.MMFFOptimizeMolecule(m)
        cls.qshape = _cdpkit_pharm_shape(m)

    def test_reference_tversky_combo_wired_and_valid(self):
        from drugex.training.scorers.rocs_cdpkit import _align_and_score_helper
        tani = _align_and_score_helper(self.qshape, self.rshape, "combo", "tanimoto")
        tvr = _align_and_score_helper(self.qshape, self.rshape, "combo", "tversky_ref")
        self.assertGreater(tvr, 0.0)
        self.assertLessEqual(tvr, 2.0 + 1e-6)            # combo range [0,2]
        self.assertNotAlmostEqual(tvr, tani, delta=1e-3)  # variant actually threaded

    def test_reference_tversky_color_in_unit_range(self):
        from drugex.training.scorers.rocs_cdpkit import _align_and_score_helper
        tvr = _align_and_score_helper(self.qshape, self.rshape, "color", "tversky_ref")
        self.assertGreaterEqual(tvr, 0.0)
        self.assertLessEqual(tvr, 1.0 + 1e-6)            # color Tversky in [0,1]

    def test_tanimoto_default_unchanged(self):
        """Default score_variant must reproduce the existing Tanimoto path (no silent change)."""
        from drugex.training.scorers.rocs_cdpkit import _align_and_score_helper
        explicit = _align_and_score_helper(self.qshape, self.rshape, "combo", "tanimoto")
        default = _align_and_score_helper(self.qshape, self.rshape, "combo")
        self.assertAlmostEqual(explicit, default, delta=1e-9)


@unittest.skipUnless(_CDPKIT, "CDPKit not available")
@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class CDPKitScorerTverskyThreading(unittest.TestCase):
    """Task #16 — CDPKitROCSScorer must accept score_variant and thread it end-to-end."""

    def test_scorer_threads_score_variant(self):
        from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer
        from drugex.training.scorers.conformer_generators import CDPKitConformerGenerator

        cg = CDPKitConformerGenerator(max_conformers=10, max_isomers=1)
        smis = ["NCCNCCNc1ccc(C(=O)O)cc1"]
        tani = CDPKitROCSScorer(cg, references=_REF_SDF, opt_mode="combo",
                                score_variant="tanimoto", show_progress=False, n_jobs=1).getScores(smis)[0, 0]
        tvr = CDPKitROCSScorer(cg, references=_REF_SDF, opt_mode="combo",
                               score_variant="tversky_ref", show_progress=False, n_jobs=1).getScores(smis)[0, 0]
        self.assertGreater(tvr, 0.0)
        self.assertNotAlmostEqual(tani, tvr, delta=1e-3)


class RDKitTverskyMath(unittest.TestCase):
    """Task #18 — RDKit Tversky on the NATIVE basis via _tversky_from_tanimoto (recovers the
    native overlap from rdShapeAlign's own reported Tanimoto + native sov/sof self-overlaps)."""

    def test_alpha_beta_one_roundtrips_to_tanimoto(self):
        """THE faithfulness proof: Tversky(a=b=1) must reproduce the input Tanimoto exactly,
        showing the recovery is on rdShapeAlign's exact native basis."""
        from drugex.training.scorers.rocs_rdkit import _tversky_from_tanimoto
        for T, sA, sB in [(0.30, 85.0, 482.0), (0.60, 100.0, 100.0), (0.0, 50.0, 50.0), (0.90, 200.0, 50.0)]:
            self.assertAlmostEqual(_tversky_from_tanimoto(T, sA, sB, 1.0, 1.0), T, places=9)

    def test_reference_tversky_forgives_larger_fit(self):
        """small ref (sA) vs big fit (sB): reference-Tversky (a=.95,b=.05) exceeds Tanimoto."""
        from drugex.training.scorers.rocs_rdkit import _tversky_from_tanimoto
        T, sA, sB = 0.30, 85.0, 482.0
        self.assertGreater(_tversky_from_tanimoto(T, sA, sB, 0.95, 0.05), T)

    def test_zero_tanimoto_gives_zero(self):
        from drugex.training.scorers.rocs_rdkit import _tversky_from_tanimoto
        self.assertEqual(_tversky_from_tanimoto(0.0, 50.0, 50.0, 0.95, 0.05), 0.0)


@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class RDKitScorerTversky(unittest.TestCase):
    """Task #18 — RDKitROCSScorer must accept score_variant, thread it, and validate it."""

    def _cg(self):
        from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
        return RDKitConformerGenerator(max_conformers=10, max_isomers=1)

    def test_scorer_threads_score_variant_end_to_end(self):
        from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
        smis = ["NCCNCCNc1ccc(C(=O)O)cc1"]
        tani = RDKitROCSScorer(self._cg(), references=_REF_SDF, score_type="color",
                               score_variant="tanimoto", show_progress=False, n_jobs=1).getScores(smis)[0, 0]
        tvr = RDKitROCSScorer(self._cg(), references=_REF_SDF, score_type="color",
                              score_variant="tversky_ref", show_progress=False, n_jobs=1).getScores(smis)[0, 0]
        self.assertGreaterEqual(tvr, 0.0)
        self.assertLessEqual(tvr, 1.0 + 1e-6)
        self.assertNotAlmostEqual(tani, tvr, delta=1e-4)

    def test_invalid_score_variant_raises(self):
        from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
        with self.assertRaises(ValueError):
            RDKitROCSScorer(self._cg(), references=_REF_SDF, score_variant="tversky", show_progress=False)


@unittest.skipUnless(_CDPKIT, "CDPKit not available")
@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class CDPKitArgValidation(unittest.TestCase):
    """Code-review #1 — invalid score_variant/opt_mode must raise (not silently score
    Tanimoto), matching OpenEyeROCSScorer which raises on a bad optimization_mode."""

    def _cg(self):
        from drugex.training.scorers.conformer_generators import CDPKitConformerGenerator
        return CDPKitConformerGenerator(max_conformers=2, max_isomers=1)

    def test_invalid_score_variant_raises(self):
        from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer
        with self.assertRaises(ValueError):  # "tversky" (missing _ref) must NOT silently == tanimoto
            CDPKitROCSScorer(self._cg(), references=_REF_SDF, score_variant="tversky", show_progress=False)

    def test_invalid_opt_mode_raises(self):
        from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer
        with self.assertRaises(ValueError):
            CDPKitROCSScorer(self._cg(), references=_REF_SDF, opt_mode="bogus", show_progress=False)


@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class GetKeyReflectsFlags(unittest.TestCase):
    """Code-review C (#19) — getKey() must reflect score_variant / inject_color / opt_mode so
    two scorers differing only by a flag get DISTINCT keys (no column collision). Defaults
    (tanimoto / combo / inject_color=False) keep the legacy keys unchanged."""

    def test_rdkit_default_key_unchanged_and_variant_distinct(self):
        from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
        from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
        cg = RDKitConformerGenerator(max_conformers=2, max_isomers=1)
        default = RDKitROCSScorer(cg, references=_REF_SDF, show_progress=False).getKey()[0]
        tvr = RDKitROCSScorer(cg, references=_REF_SDF, score_variant="tversky_ref", show_progress=False).getKey()[0]
        inj = RDKitROCSScorer(cg, references=_REF_SDF, inject_color=True, show_progress=False).getKey()[0]
        self.assertNotIn("tversky", default)
        self.assertIn("tversky_ref", tvr)
        self.assertIn("inject", inj)
        self.assertNotEqual(default, tvr)

    @unittest.skipUnless(_CDPKIT, "CDPKit not available")
    def test_cdpkit_variant_and_mode_distinct(self):
        from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer
        from drugex.training.scorers.conformer_generators import CDPKitConformerGenerator
        cg = CDPKitConformerGenerator(max_conformers=2, max_isomers=1)
        default = CDPKitROCSScorer(cg, references=_REF_SDF, show_progress=False).getKey()[0]
        tvr = CDPKitROCSScorer(cg, references=_REF_SDF, opt_mode="color",
                               score_variant="tversky_ref", show_progress=False).getKey()[0]
        self.assertNotIn("tversky", default)
        self.assertIn("tversky_ref", tvr)
        self.assertIn("color", tvr)


try:
    from openeye import oechem as _oechem
    _OE = _oechem.OEChemIsLicensed()
except Exception:
    _OE = False
_ROCS_BIN = os.path.join(_REPO_ROOT, "ccr2_gen/oeye/current_apps/apps/openeye/bin/rocs")


@unittest.skipUnless(_OE, "OpenEye toolkit not licensed/available")
@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class OpenEyeTversky(unittest.TestCase):
    """Task #16/#18 — OpenEyeROCSScorer must map (mode, score_variant) to the correct ROCS
    report column (verified names) and validate the variant."""

    def _scorer(self, mode, variant):
        from drugex.training.scorers.rocs_openeye import OpenEyeROCSScorer
        from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
        cg = RDKitConformerGenerator(max_conformers=3, max_isomers=1)
        return OpenEyeROCSScorer(cg, references={"ref": _REF_SDF}, optimization_mode=mode,
                                 score_variant=variant, binary_path=_ROCS_BIN, show_progress=False)

    def test_column_selection_matches_verified_report_names(self):
        self.assertEqual(self._scorer("combo", "tanimoto")._score_column, "TanimotoCombo")
        self.assertEqual(self._scorer("color", "tanimoto")._score_column, "ColorTanimoto")
        self.assertEqual(self._scorer("color", "tversky_ref")._score_column, "RefColorTversky")
        self.assertEqual(self._scorer("combo", "tversky_ref")._score_column, "RefTverskyCombo")
        self.assertEqual(self._scorer("shape", "tversky_fit")._score_column, "FitTversky")
        self.assertEqual(self._scorer("combo", "tversky_fit")._score_column, "FitTverskyCombo")

    def test_invalid_score_variant_raises(self):
        with self.assertRaises(ValueError):
            self._scorer("combo", "tversky")  # missing _ref

    def test_shape_only_uses_shape_tversky_column(self):
        """Code-review A: shape_only=True (without optimization_mode) + Tversky must read the
        SHAPE Tversky column (present in a -shapeonly report), NOT RefTverskyCombo (absent)."""
        from drugex.training.scorers.rocs_openeye import OpenEyeROCSScorer
        from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
        cg = RDKitConformerGenerator(max_conformers=3, max_isomers=1)
        s = OpenEyeROCSScorer(cg, references={"ref": _REF_SDF}, shape_only=True,
                              score_variant="tversky_ref", binary_path=_ROCS_BIN, show_progress=False)
        self.assertEqual(s._score_column, "RefTversky")

    def test_parse_results_skips_nan_cells(self):
        """Code-review B: a NaN/blank score cell must not poison the per-molecule running max."""
        import tempfile
        from drugex.training.scorers.rocs_openeye import OpenEyeROCSScorer
        from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
        cg = RDKitConformerGenerator(max_conformers=3, max_isomers=1)
        s = OpenEyeROCSScorer(cg, references={"ref": _REF_SDF}, optimization_mode="combo",
                              binary_path=_ROCS_BIN, show_progress=False)
        tsv = tempfile.mktemp(suffix=".tsv")
        with open(tsv, "w") as f:  # mol_0: a NaN conformer FIRST, then a good one
            f.write("Name\tTanimotoCombo\n")
            f.write("mol_0+0\t\n")          # blank -> NaN
            f.write("mol_0+1\t1.234\n")
        scores = s._parse_results(tsv)
        os.unlink(tsv)
        self.assertIn(0, scores)
        self.assertAlmostEqual(scores[0], 1.234, places=3)  # not NaN, not 0

    @unittest.skipUnless(os.path.exists(_ROCS_BIN), f"rocs binary not found: {_ROCS_BIN}")
    def test_integration_tversky_ref_differs_from_tanimoto(self):
        os.environ["LD_LIBRARY_PATH"] = (
            os.path.join(_REPO_ROOT, "ccr2_gen/oeye/runtime_libs") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )
        _lic = os.path.join(_REPO_ROOT, "ccr2_gen/oe_license.txt")
        if os.path.exists(_lic):
            os.environ.setdefault("OE_LICENSE", _lic)
        smis = ["NCCNCCNc1ccc(C(=O)O)cc1"]
        tani = self._scorer("color", "tanimoto").getScores(smis)[0, 0]
        tvr = self._scorer("color", "tversky_ref").getScores(smis)[0, 0]
        self.assertGreaterEqual(tvr, 0.0)
        self.assertNotAlmostEqual(tani, tvr, delta=1e-4)  # the Tversky column was actually parsed


if __name__ == "__main__":
    unittest.main(verbosity=2)
