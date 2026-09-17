"""Characterization + behavior tests for the ROCS shape/color optimization mode switch.

Run:  ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests_color_modes

Task 0 (this commit) = the SAFETY NET: lock the CURRENT behavior of
rocs_rdkit._score_single_reference before wiring `opt_param` (Task B), so any
change is provable. These characterization tests assert the current numbers
(RDKit shape-only optimization, opt_param defaults to 1.0).
"""
import os
import shutil
import unittest

from rdkit import Chem

try:
    from rdkit.Chem import rdShapeAlign  # noqa: F401
    _RDSHAPEALIGN = True
except ImportError:
    _RDSHAPEALIGN = False

from drugex.training.scorers.rocs_rdkit import _score_single_reference

# Reference ligands shipped with the CCR2 project (5 rigid, single-conformer mols).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_REF_SDF = os.path.join(
    _REPO_ROOT,
    "tutorial/advanced/rocs/rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf",
)

# Golden values for _score_single_reference(ref0, ref1). Current default = opt_param=1.0
# (shape-only optimization); combo == shape + color.
# Re-baselined 2026-06-09 after the issue-#8513 pose-invariance guard (Task #17): the guard
# canonicalizes to the inertial frame + tries principal-axis sign starts and keeps the best
# overlay, so it escapes the pose-dependent local optimum and shifts these rigid-reference
# scores slightly (color +0.0016, combo -0.0006 vs the pre-guard 2026-06-05 / RDKit 2025.09.6
# values of combo 0.854252 / shape 0.731149 / color 0.123103). The shift IS the fix.
_GOLD = {"TanimotoCombo": 0.853650, "shape": 0.731162, "color": 0.124658}
_TOL = 1e-3


@unittest.skipUnless(_RDSHAPEALIGN, "rdShapeAlign not available")
@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class CurrentRDKitBehavior(unittest.TestCase):
    """SAFETY NET — locks current shape-only-optimization behavior before Task B."""

    @classmethod
    def setUpClass(cls):
        mols = [m for m in Chem.SDMolSupplier(_REF_SDF, removeHs=False) if m]
        assert len(mols) >= 2, "need at least 2 reference molecules"
        cls.ref0, cls.ref1 = mols[0], mols[1]

    def test_combo_matches_golden(self):
        v = _score_single_reference(self.ref0, self.ref1, "TanimotoCombo", True)
        self.assertAlmostEqual(v, _GOLD["TanimotoCombo"], delta=_TOL)

    def test_shape_matches_golden(self):
        v = _score_single_reference(self.ref0, self.ref1, "shape", True)
        self.assertAlmostEqual(v, _GOLD["shape"], delta=_TOL)

    def test_color_matches_golden(self):
        v = _score_single_reference(self.ref0, self.ref1, "color", True)
        self.assertAlmostEqual(v, _GOLD["color"], delta=_TOL)

    def test_combo_equals_shape_plus_color_on_rigid_refs(self):
        """On rigid single-conformer references the shape / combo / color optimization
        objectives converge to (nearly) the same pose, so combo ~= shape + color regardless
        of opt_param. This invariance is *why* Task B (driving opt_param from score mode)
        leaves rigid-reference scores essentially unchanged — the behavior change is on
        flexible molecules (see ColorOptimizationModes).

        Tolerance is 3e-3 (was 1e-6): since the issue-#8513 guard (Task #17) optimizes each
        mode from its OWN best principal-axis sign start, shape-opt and color-opt poses on
        rigid refs no longer coincide to the bit, but still agree to ~2.2e-3 (measured)."""
        s = _score_single_reference(self.ref0, self.ref1, "shape", True)
        c = _score_single_reference(self.ref0, self.ref1, "color", True)
        combo = _score_single_reference(self.ref0, self.ref1, "TanimotoCombo", True)
        self.assertAlmostEqual(s + c, combo, delta=3e-3)


def _build_flexible_probe():
    """Deterministic flexible probe with real color headroom vs ref0.

    ether_amine (8 rotatable bonds): color@color-opt exceeds color@shape-opt by ~0.10,
    so it reveals whether color-mode actually optimizes the pose for color.
    """
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles("CN(C)CCCOc1ccc(C(=O)O)cc1"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    AllChem.EmbedMultipleConfs(m, numConfs=20, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(m)
    return m


@unittest.skipUnless(_RDSHAPEALIGN, "rdShapeAlign not available")
@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class ColorOptimizationModes(unittest.TestCase):
    """Task B — score_type must drive the alignment OPTIMIZATION (opt_param), not
    just which component is returned. color-mode must co-optimize the pose for color."""

    @classmethod
    def setUpClass(cls):
        cls.ref0 = [m for m in Chem.SDMolSupplier(_REF_SDF, removeHs=False) if m][0]
        cls.probe = _build_flexible_probe()

    def _best_color_at_shape_optimized_pose(self):
        """Independent baseline: best ColorTanimoto at the SHAPE-optimized pose (opt_param=1.0)."""
        from rdkit.Chem import rdShapeAlign

        best = 0.0
        for pc in self.probe.GetConformers():
            for rc in self.ref0.GetConformers():
                res = rdShapeAlign.AlignMol(
                    self.ref0, Chem.Mol(self.probe), rc.GetId(), pc.GetId(), True, 1.0
                )
                best = max(best, res[1])
        return best

    def test_color_mode_optimizes_pose_for_color(self):
        """color-mode must optimize FOR color -> exceed the color at the shape-opt pose."""
        base = self._best_color_at_shape_optimized_pose()
        color_mode = _score_single_reference(self.probe, self.ref0, "color", True)
        self.assertGreater(
            color_mode,
            base + 1e-3,
            msg=f"color-mode ({color_mode:.4f}) must exceed color@shape-pose ({base:.4f}); "
            "score_type must drive opt_param=0.0 for color mode",
        )


try:
    import CDPL.Shape as _CDPLShape  # noqa: F401  (rdkit already imported above -> safe order)
    import CDPL.Chem as _CDPLChem  # noqa: F401
    import CDPL.Pharm as _CDPLPharm  # noqa: F401

    _CDPKIT = True
except ImportError:
    _CDPKIT = False


def _cdpkit_pharm_shape(rdkit_mol):
    """Build a CDPKit pharmacophore (colored) Gaussian shape from an RDKit mol w/ a conformer."""
    import tempfile

    sdf = tempfile.mktemp(suffix=".sdf")
    w = Chem.SDWriter(sdf)
    w.write(rdkit_mol)
    w.close()
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
class CDPKitColorSeeding(unittest.TestCase):
    """Task C — CDPKit color mode must seed alignment starts from color-feature centers
    (CDPKit has no color gradient; seeding + selection is the closest approach to OE)."""

    @classmethod
    def setUpClass(cls):
        from rdkit.Chem import AllChem

        ref0 = [m for m in Chem.SDMolSupplier(_REF_SDF, removeHs=False) if m][0]
        cls.rshape = _cdpkit_pharm_shape(ref0)
        # triamine_acid: a probe where color-center seeding demonstrably helps (+0.04 color)
        m = Chem.AddHs(Chem.MolFromSmiles("NCCNCCNc1ccc(C(=O)O)cc1"))
        p = AllChem.ETKDGv3()
        p.randomSeed = 7
        AllChem.EmbedMolecule(m, p)
        AllChem.MMFFOptimizeMolecule(m)
        cls.qshape = _cdpkit_pharm_shape(m)

    def _best_color_shape_seeded(self):
        """Baseline: best ColorTanimoto at the SHAPE-center-seeded pose (no color seeding)."""
        a = _CDPLShape.GaussianShapeAlignment()
        a.setStartGenerator(_CDPLShape.PrincipalAxesAlignmentStartGenerator())
        a.setMaxNumOptimizationIterations(20)
        a.setOptimizationStopGradient(1.0)
        a.addReferenceShape(self.rshape)
        best = 0.0
        if a.align(self.qshape):
            for i in range(a.getNumResults()):
                best = max(best, _CDPLShape.calcColorTanimotoScore(a.getResult(i)))
        return best

    def test_color_mode_uses_color_center_seeding(self):
        from drugex.training.scorers.rocs_cdpkit import _align_and_score_helper

        base = self._best_color_shape_seeded()
        color_mode = _align_and_score_helper(self.qshape, self.rshape, opt_mode="color")
        self.assertGreater(
            color_mode,
            base + 1e-3,
            msg=f"color mode ({color_mode:.4f}) must exceed color@shape-seeded pose "
            f"({base:.4f}) via color-feature-center seeding",
        )


@unittest.skipUnless(_CDPKIT, "CDPKit not available")
@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class CDPKitScorerModes(unittest.TestCase):
    """Task C2 — CDPKitROCSScorer must accept opt_mode and thread it to the worker."""

    def test_scorer_threads_opt_mode_end_to_end(self):
        from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer
        from drugex.training.scorers.conformer_generators import CDPKitConformerGenerator

        cg = CDPKitConformerGenerator(max_conformers=10, max_isomers=1)
        smis = ["NCCNCCNc1ccc(C(=O)O)cc1"]  # triamine_acid
        s = {}
        for mode in ("shape", "combo", "color"):
            scorer = CDPKitROCSScorer(
                cg, references=_REF_SDF, opt_mode=mode, show_progress=False, n_jobs=1
            )
            s[mode] = float(scorer.getScores(smis)[0, 0])
        # shape & color are Tanimoto in [0,1]; combo = shape + color can exceed 1
        self.assertLessEqual(s["shape"], 1.0 + 1e-6)
        self.assertLessEqual(s["color"], 1.0 + 1e-6)
        self.assertGreaterEqual(s["combo"], s["shape"] - 1e-6)
        # modes must produce different outputs -> opt_mode is actually threaded
        self.assertNotAlmostEqual(s["combo"], s["color"], delta=1e-3)


try:
    from openeye import oechem as _oechem

    _OE = _oechem.OEChemIsLicensed()
except Exception:
    _OE = False

_ROCS_BIN = (
    os.environ.get("ROCS_BINARY")
    or shutil.which("rocs")
    or os.path.join(_REPO_ROOT, "ccr2_gen/oeye/current_apps/apps/openeye/bin/rocs")
)


@unittest.skipUnless(_OE, "OpenEye toolkit not licensed/available")
@unittest.skipUnless(os.path.exists(_ROCS_BIN), f"rocs binary not found: {_ROCS_BIN}")
@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class OpenEyeModes(unittest.TestCase):
    """Task A — OpenEyeROCSScorer must accept optimization_mode and map it to the right
    ROCS CLI flags (shape -> -shapeonly; combo/color -> -rankby ... -optchem true)."""

    def _cmd(self, mode):
        from drugex.training.scorers.rocs_openeye import OpenEyeROCSScorer
        from drugex.training.scorers.conformer_generators import RDKitConformerGenerator

        cg = RDKitConformerGenerator(max_conformers=5, max_isomers=1)
        scorer = OpenEyeROCSScorer(
            cg, references={"ref": _REF_SDF}, optimization_mode=mode,
            binary_path=_ROCS_BIN, show_progress=False,
        )
        return scorer._build_rocs_command("q.sdf", "in.sdf", "out.tsv")

    def test_shape_mode_flags(self):
        cmd = self._cmd("shape")
        self.assertIn("-shapeonly", cmd)

    def test_combo_mode_flags(self):
        cmd = self._cmd("combo")
        self.assertNotIn("-shapeonly", cmd)
        self.assertEqual(cmd[cmd.index("-rankby") + 1], "TanimotoCombo")
        self.assertEqual(cmd[cmd.index("-optchem") + 1], "true")

    def test_color_mode_flags(self):
        cmd = self._cmd("color")
        self.assertNotIn("-shapeonly", cmd)
        self.assertEqual(cmd[cmd.index("-rankby") + 1], "ColorTanimoto")
        self.assertEqual(cmd[cmd.index("-optchem") + 1], "true")


if __name__ == "__main__":
    unittest.main(verbosity=2)
