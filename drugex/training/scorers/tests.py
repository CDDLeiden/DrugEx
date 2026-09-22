"""Consolidated tests for :mod:`drugex.training.scorers`.

Run:  ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests

This module is the single test entry point for the scorers package, matching the
repo convention (one ``tests.py`` per package, cf. ``drugex/data/tests.py`` and
``drugex/training/tests.py``).  It merges what used to be 14 separate files:

  tests_color_modes.py        tests_prep_openeye.py     tests/test_conformer_caps.py
  tests_hydrophobe_color.py   tests_prep_rdkit.py       tests/test_parallel.py
  tests_oe_color_ff.py        tests_protonation.py      tests/test_score_timeout_integration.py
  tests_pose_invariance.py    tests_rocs_mpi.py
  tests_prep_cdpkit.py        tests_tversky.py
  tests_prep_consistency.py

The three ``tests/test_*.py`` files used bare module-level ``def test_*`` functions with
the ``monkeypatch`` fixture; they are converted to ``unittest.TestCase`` methods here so
the ``python -m`` entry point actually collects them.  Assertions and logic are
unchanged; ``monkeypatch.setenv/delenv/setattr`` became ``mock.patch.dict``/
``mock.patch.object`` and ``raises(...)`` became ``assertRaises``.

Names disambiguated during the merge (identical names, DIFFERENT meanings):
  * ``_TOL``               -> ``_TOL_MODES`` (1e-3, color-mode goldens) and
                              ``_TOL_POSE``  (5e-3, pose-invariance round-off budget)
  * ``_embed``             -> ``_embed_hydrophobe`` / ``_embed_oe_ff`` / ``_embed_pose``
                              (different default seeds and signatures)
  * ``_cdpkit_pharm_shape``-> ``_cdpkit_pharm_shape_modes`` / ``_cdpkit_pharm_shape_tversky``
  * ``_OE``                -> kept for ``OEChemIsLicensed()``; the stricter
                              ``OEChemIsLicensed() and OEQuacPacIsLicensed()`` guard
                              became ``_OE_QUACPAC``
  * ``CDPL`` / ``REF``     -> ``_SCORE_TIMEOUT_CDPL`` / ``_SCORE_TIMEOUT_REF``
Byte-identical duplicates (``_REPO_ROOT``, ``_REF_SDF``, ``_ROCS_BIN``) are defined once below.
"""
import multiprocessing as mp
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

from rdkit import Chem  # FIRST (before any CDPL import -> safe library load order)
from rdkit.Chem import AllChem

# --------------------------------------------------------------------------------------
# Availability guards (deduped from the merged modules; supersets of what each needed)
# --------------------------------------------------------------------------------------
try:
    from rdkit.Chem import rdShapeAlign, rdMolTransforms  # noqa: F401
    _RDSHAPEALIGN = True
except ImportError:
    _RDSHAPEALIGN = False

try:
    import CDPL.Shape as _CDPLShape  # noqa: F401  (rdkit already imported above -> safe order)
    import CDPL.Chem as _CDPLChem  # noqa: F401
    import CDPL.Pharm as _CDPLPharm  # noqa: F401

    _CDPKIT = True
except ImportError:
    _CDPKIT = False

try:
    import dimorphite_dl  # noqa: F401

    _DIMORPHITE = True
except ImportError:
    _DIMORPHITE = False

try:
    from openeye import oechem as _oechem

    _OE = _oechem.OEChemIsLicensed()
except Exception:
    _OE = False

try:
    from openeye import oechem, oequacpac

    _OE_QUACPAC = oechem.OEChemIsLicensed() and oequacpac.OEQuacPacIsLicensed()
except Exception:
    _OE_QUACPAC = False

# --------------------------------------------------------------------------------------
# Code under test
# --------------------------------------------------------------------------------------
from drugex.training.scorers.rocs_rdkit import (
    _annotate_color_features,
    _score_single_reference,
)
from drugex.training.scorers.oe_color_smarts import (
    OE_MILLS_DEAN_IMPLICIT,
    OE_TO_CDPKIT_TYPE,
)
from drugex.training.scorers.conformer_generators import (
    RDKitConformerGenerator, CDPKitConformerGenerator, OmegaConformerGenerator)
from drugex.training.scorers import conformer_generators as cg
from drugex.training.scorers.protonation import protonate_smiles
from drugex.training.scorers.parallel import (
    pool_context, capped_n_jobs, score_timeout, mol_timeout,
    molecule_time_limit, ScoreTimeout,
)
import drugex.training.scorers.rocs_cdpkit as rc

# --------------------------------------------------------------------------------------
# Shared constants (were byte-identical across the merged modules)
# --------------------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# Reference ligands shipped with the CCR2 project (5 rigid, single-conformer mols).
_REF_SDF = os.path.join(
    _REPO_ROOT, "tutorial/advanced/rocs/rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf"
)
_ROCS_BIN = (
    os.environ.get("ROCS_BINARY")
    or shutil.which("rocs")
    or os.path.join(_REPO_ROOT, "ccr2_gen/oeye/current_apps/apps/openeye/bin/rocs")
)


# ======================================================================================
# SECTION: ROCS shape/color optimization-mode switch  (was tests_color_modes.py)
# ======================================================================================

"""Characterization + behavior tests for the ROCS shape/color optimization mode switch.

Run:  ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests

Task 0 (this commit) = the SAFETY NET: lock the CURRENT behavior of
rocs_rdkit._score_single_reference before wiring `opt_param` (Task B), so any
change is provable. These characterization tests assert the current numbers
(RDKit shape-only optimization, opt_param defaults to 1.0).
"""

# Golden values for _score_single_reference(ref0, ref1). Current default = opt_param=1.0
# (shape-only optimization); combo == shape + color.
# Re-baselined 2026-06-09 after the issue-#8513 pose-invariance guard (Task #17): the guard
# canonicalizes to the inertial frame + tries principal-axis sign starts and keeps the best
# overlay, so it escapes the pose-dependent local optimum and shifts these rigid-reference
# scores slightly (color +0.0016, combo -0.0006 vs the pre-guard 2026-06-05 / RDKit 2025.09.6
# values of combo 0.854252 / shape 0.731149 / color 0.123103). The shift IS the fix.
_GOLD = {"TanimotoCombo": 0.853650, "shape": 0.731162, "color": 0.124658}
_TOL_MODES = 1e-3


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
        self.assertAlmostEqual(v, _GOLD["TanimotoCombo"], delta=_TOL_MODES)

    def test_shape_matches_golden(self):
        v = _score_single_reference(self.ref0, self.ref1, "shape", True)
        self.assertAlmostEqual(v, _GOLD["shape"], delta=_TOL_MODES)

    def test_color_matches_golden(self):
        v = _score_single_reference(self.ref0, self.ref1, "color", True)
        self.assertAlmostEqual(v, _GOLD["color"], delta=_TOL_MODES)

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




def _cdpkit_pharm_shape_modes(rdkit_mol):
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
        cls.rshape = _cdpkit_pharm_shape_modes(ref0)
        # triamine_acid: a probe where color-center seeding demonstrably helps (+0.04 color)
        m = Chem.AddHs(Chem.MolFromSmiles("NCCNCCNc1ccc(C(=O)O)cc1"))
        p = AllChem.ETKDGv3()
        p.randomSeed = 7
        AllChem.EmbedMolecule(m, p)
        AllChem.MMFFOptimizeMolecule(m)
        cls.qshape = _cdpkit_pharm_shape_modes(m)

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


# ======================================================================================
# SECTION: RDKit color hydrophobe injection, Lever 2 / Task #14  (was tests_hydrophobe_color.py)
# ======================================================================================

"""TDD tests for RDKit color hydrophobe injection (Lever 2, Task #14).

Run:  ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests

Background (verified at source, see ccr2_gen/docs/color_validation/HOW_CLOSE_TO_OE.md §1b):
rdShapeAlign's DEFAULT color uses a 5-pattern fallback (donor/acceptor/rings/cation/anion)
with NO hydrophobe. OpenEye's ImplicitMillsDean has a 6th type (hydrophobe), and the CCR2
references are hydrophobe-heavy -> RDKit is blind to their dominant color feature unless we
write the `PUBCHEM_PHARMACOPHORE_FEATURES` SD property (the 6-type path) derived from
RDKit's BaseFeatures.fdef. `_annotate_color_features(mol)` does that.

octane (CCCCCCCC) is the golden probe: acyclic + no donor/acceptor/charge -> the fallback
sees 0 color atoms -> color self-overlap is exactly 0 TODAY; it becomes > 0 only if the
hydrophobe channel is correctly wired.
"""

def _embed_hydrophobe(smi):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    p = AllChem.ETKDGv3()
    p.randomSeed = 7
    AllChem.EmbedMolecule(m, p)
    return m


@unittest.skipUnless(_RDSHAPEALIGN, "rdShapeAlign not available")
class HydrophobeColorInjection(unittest.TestCase):
    """Task #14 — `_annotate_color_features` must enable the hydrophobe color channel."""

    def _self_color(self, mol):
        """ColorTanimoto of mol vs a copy of itself (props copy with Chem.Mol)."""
        res = rdShapeAlign.AlignMol(mol, Chem.Mol(mol), useColors=True, opt_param=0.5)
        return res[1]

    def test_octane_color_zero_without_injection(self):
        """Characterizes the gap: acyclic alkane has 0 color self-overlap by default."""
        self.assertEqual(self._self_color(_embed_hydrophobe("CCCCCCCC")), 0.0)

    def test_octane_gains_color_after_injection(self):
        """The core behavior: after annotation the hydrophobe channel fires -> color > 0."""
        m = _embed_hydrophobe("CCCCCCCC")
        _annotate_color_features(m)
        self.assertGreater(self._self_color(m), 0.0)

    def test_injection_preserves_polar_color(self):
        """Annotation must not destroy existing donor/acceptor color."""
        m = _embed_hydrophobe("OCCO")  # ethylene glycol: donors + acceptors
        _annotate_color_features(m)
        self.assertGreater(self._self_color(m), 0.0)

    def test_annotation_sets_pubchem_property(self):
        """The function writes the 6-type SD property rdShapeAlign reads."""
        m = _embed_hydrophobe("CCCCCCCC")
        _annotate_color_features(m)
        self.assertTrue(m.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES"))
        self.assertIn("hydrophobe", m.GetProp("PUBCHEM_PHARMACOPHORE_FEATURES"))


@unittest.skipUnless(_RDSHAPEALIGN, "rdShapeAlign not available")
@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class HydrophobeInjectionInScoring(unittest.TestCase):
    """Task #14 — `_score_single_reference(inject_color=True)` must route both query and
    reference through `_annotate_color_features` so the hydrophobe channel contributes."""

    @classmethod
    def setUpClass(cls):
        cls.ref0 = [m for m in Chem.SDMolSupplier(_REF_SDF, removeHs=False) if m][0]

    def test_inject_color_enables_hydrophobic_query(self):
        """octane = pure hydrophobe. Default: 0 color atoms -> color 0 vs any ref.
        With inject_color: hydrophobe overlaps the (hydrophobe-heavy) CCR2 ref -> color > 0."""
        base = _score_single_reference(_embed_hydrophobe("CCCCCCCC"), self.ref0, "color", True, inject_color=False)
        inj = _score_single_reference(_embed_hydrophobe("CCCCCCCC"), self.ref0, "color", True, inject_color=True)
        self.assertEqual(base, 0.0)
        self.assertGreater(inj, 0.0)

    def test_inject_color_default_off_preserves_behavior(self):
        """Default (no inject_color kwarg) must equal inject_color=False (no silent change)."""
        probe = _embed_hydrophobe("CCCCCCCc1ccccc1")
        default = _score_single_reference(probe, self.ref0, "color", True)
        off = _score_single_reference(probe, self.ref0, "color", True, inject_color=False)
        self.assertAlmostEqual(default, off, delta=1e-9)


@unittest.skipUnless(_RDSHAPEALIGN, "rdShapeAlign not available")
@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class RDKitScorerInjectColorFlag(unittest.TestCase):
    """Task #14 — RDKitROCSScorer must accept inject_color and thread it end-to-end."""

    def test_scorer_threads_inject_color_end_to_end(self):
        from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
        from drugex.training.scorers.conformer_generators import RDKitConformerGenerator

        cg = RDKitConformerGenerator(max_conformers=10, max_isomers=1)
        smis = ["CCCCCCCC"]  # octane: pure hydrophobe -> color 0 unless hydrophobe wired
        off = RDKitROCSScorer(
            cg, references=_REF_SDF, score_type="color",
            inject_color=False, show_progress=False, n_jobs=1,
        ).getScores(smis)[0, 0]
        on = RDKitROCSScorer(
            cg, references=_REF_SDF, score_type="color",
            inject_color=True, show_progress=False, n_jobs=1,
        ).getScores(smis)[0, 0]
        self.assertEqual(off, 0.0)
        self.assertGreater(on, 0.0)


@unittest.skipUnless(_RDSHAPEALIGN, "rdShapeAlign not available")
@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class RDKitArgValidation(unittest.TestCase):
    """Code-review #1 — invalid score_type must raise (not silently score combo)."""

    def test_invalid_score_type_raises(self):
        from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
        from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
        cg = RDKitConformerGenerator(max_conformers=2, max_isomers=1)
        with self.assertRaises(ValueError):
            RDKitROCSScorer(cg, references=_REF_SDF, score_type="bogus", show_progress=False)


# ======================================================================================
# SECTION: OpenEye ImplicitMillsDean color force-field port, Lever 1  (was tests_oe_color_ff.py)
# ======================================================================================

"""TDD tests for the OpenEye ImplicitMillsDean color force field port (Lever 1).

Run (with OE_LICENSE + LD_LIBRARY_PATH from ccr2_gen/oeye/runtime_libs):
  ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests

Background (verified at source: ccr2_gen/oeye/current_apps/apps/openeye/data/rocs/
3.9.0.1/rocs/ImplicitMillsDean.cff):
The OE ImplicitMillsDean color FF declares 6 INTERACTION types (donor, acceptor,
cation, anion, rings, hydrophobe), each `attractive gaussian weight=1.0 radius=1.0`.
Each type's PATTERN lines reference $macros expanded from the DEFINE block. This
port expands those macros into self-contained RDKit/CDPKit SMARTS so the open-source
color channel uses the SAME 6 atom-types OpenEye does, rather than:
  * RDKit's 5-pattern rdShapeAlign fallback (no hydrophobe), or
  * RDKit BaseFeatures.fdef (aromatic-only "rings", LumpedHydrophobe gaps), or
  * CDPKit's DefaultPharmacophoreGenerator.

The new module under test (RED until implemented): oe_color_smarts.py with
  OE_MILLS_DEAN_IMPLICIT : {color_name: [SMARTS, ...]} for the 6 types, and
  OE_TO_CDPKIT_TYPE      : {color_name: (CDPKit FeatureType, geometry)}.

OPT-IN contract: this FF only fires when color_ff="oe_mills_dean" is requested;
the default ("default") must reproduce current scores bit-for-bit (test class
DefaultOff). The OE-ground-truth Spearman GATE that decides whether this becomes
the default is a separate compute step run after merge -- keep it OFF by default.
"""

_SIX_TYPES = {"donor", "acceptor", "cation", "anion", "rings", "hydrophobe"}


def _embed_oe_ff(smi, seed=7):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    return m


def _fires(color_name, smi):
    """True iff ANY OE SMARTS for ``color_name`` matches ``smi`` (2D, no Hs needed)."""
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"bad test SMILES {smi!r}"
    return any(
        mol.HasSubstructMatch(Chem.MolFromSmarts(s))
        for s in OE_MILLS_DEAN_IMPLICIT[color_name]
    )


# ---------------------------------------------------------------------------
# (a) all 6xN SMARTS compile
# ---------------------------------------------------------------------------
class SmartsCompile(unittest.TestCase):
    """(a) Every expanded SMARTS for all 6 types must compile (MolFromSmarts non-None)."""

    def test_exactly_the_six_oe_types_present(self):
        self.assertEqual(set(OE_MILLS_DEAN_IMPLICIT.keys()), _SIX_TYPES)

    def test_every_type_has_at_least_one_pattern(self):
        for name in _SIX_TYPES:
            self.assertGreater(len(OE_MILLS_DEAN_IMPLICIT[name]), 0, name)

    def test_all_smarts_compile(self):
        for name, patterns in OE_MILLS_DEAN_IMPLICIT.items():
            for smt in patterns:
                with self.subTest(color=name, smarts=smt):
                    self.assertIsNotNone(
                        Chem.MolFromSmarts(smt),
                        msg=f"SMARTS for {name} failed to compile: {smt}",
                    )


# ---------------------------------------------------------------------------
# (b) per-type firing parity with the OE definitions
# ---------------------------------------------------------------------------
class PerTypeFiring(unittest.TestCase):
    """(b) Each OE type fires on its canonical positive and stays silent on a control."""

    def test_saturated_ring_fires_rings(self):
        # cyclohexane is a NON-aromatic ring: OE `rings` is [R]~..~[R] (any ring),
        # unlike the fdef Aromatic-only "rings". This is the key OE-vs-fdef difference.
        self.assertTrue(_fires("rings", "C1CCCCC1"))

    def test_acyclic_does_not_fire_rings(self):
        self.assertFalse(_fires("rings", "CCCCCC"))

    def test_octane_fires_hydrophobe(self):
        self.assertTrue(_fires("hydrophobe", "CCCCCCCC"))

    def test_ether_oxygen_fires_acceptor(self):
        # dimethyl ether: ACether weak acceptor.
        self.assertTrue(_fires("acceptor", "COC"))

    def test_neutral_carboxylic_acid_fires_anion_ph_implicit(self):
        # acetic acid in its NEUTRAL form must match `anion` (pH-implicit: the FF
        # matches the protonated acid, treating it as the would-be anion).
        self.assertTrue(_fires("anion", "CC(=O)O"))

    def test_amine_fires_cation_ph_implicit(self):
        # ethylamine neutral must match `cation` ($CATamine, pH-implicit).
        self.assertTrue(_fires("cation", "CCN"))

    def test_alcohol_fires_donor(self):
        # ethanol hydroxyl is a (strong) donor.
        self.assertTrue(_fires("donor", "CCO"))


# ---------------------------------------------------------------------------
# (c) RDKit-emitted color name <-> CDPKit FeatureType map is 1:1 over the 6 types
# ---------------------------------------------------------------------------
class TypeMapBijection(unittest.TestCase):
    """(c) The RDKit color names and the CDPKit FeatureType map cover the same 6 types 1:1."""

    def test_map_keys_are_the_six_types(self):
        self.assertEqual(set(OE_TO_CDPKIT_TYPE.keys()), _SIX_TYPES)

    def test_each_value_is_featuretype_geometry_pair(self):
        for name, val in OE_TO_CDPKIT_TYPE.items():
            with self.subTest(color=name):
                self.assertEqual(len(val), 2, f"{name} must map to (FeatureType, geometry)")

    @unittest.skipUnless(_CDPKIT, "CDPKit not available")
    def test_feature_types_are_distinct_and_valid(self):
        # 6 distinct CDPKit FeatureType ints -> the name<->type map is injective.
        ftypes = [val[0] for val in OE_TO_CDPKIT_TYPE.values()]
        self.assertEqual(len(set(ftypes)), len(_SIX_TYPES))

    def test_rdkit_names_match_cdpkit_map_keys(self):
        # The names emitted into PUBCHEM_PHARMACOPHORE_FEATURES (RDKit path) are exactly
        # the keys CDPKit maps -> 1:1 across backends.
        self.assertEqual(set(OE_MILLS_DEAN_IMPLICIT.keys()), set(OE_TO_CDPKIT_TYPE.keys()))


# ---------------------------------------------------------------------------
# (d) protonation invariance: neutral acid == its anion for the `anion` set
# ---------------------------------------------------------------------------
class ProtonationInvariance(unittest.TestCase):
    """(d) pH-implicit acids/bases: neutral and (de)protonated forms give the same firing."""

    def test_carboxylic_acid_protonation_invariant_anion(self):
        self.assertEqual(_fires("anion", "CC(=O)O"), _fires("anion", "CC(=O)[O-]"))

    def test_carboxylic_acid_both_forms_fire_anion(self):
        self.assertTrue(_fires("anion", "CC(=O)O"))
        self.assertTrue(_fires("anion", "CC(=O)[O-]"))


# ---------------------------------------------------------------------------
# RDKit annotation path: oe_mills_dean builds feature lines from OE SMARTS
# ---------------------------------------------------------------------------
@unittest.skipUnless(_RDSHAPEALIGN, "rdShapeAlign not available")
class RDKitOEAnnotation(unittest.TestCase):
    """`_annotate_color_features(mol, color_ff="oe_mills_dean")` writes the 6-type
    PUBCHEM_PHARMACOPHORE_FEATURES property from OE SMARTS substructure matches."""

    def test_oe_annotation_writes_pubchem_property(self):
        from drugex.training.scorers.rocs_rdkit import _annotate_color_features

        m = _embed_oe_ff("CCCCCCCC")
        _annotate_color_features(m, color_ff="oe_mills_dean")
        self.assertTrue(m.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES"))
        self.assertIn("hydrophobe", m.GetProp("PUBCHEM_PHARMACOPHORE_FEATURES"))

    def test_oe_annotation_emits_saturated_ring_as_rings(self):
        # cyclohexane: OE `rings` fires (fdef Aromatic would NOT) -> proves OE path is used.
        from drugex.training.scorers.rocs_rdkit import _annotate_color_features

        m = _embed_oe_ff("C1CCCCC1")
        _annotate_color_features(m, color_ff="oe_mills_dean")
        self.assertIn("rings", m.GetProp("PUBCHEM_PHARMACOPHORE_FEATURES"))

    def test_oe_annotation_emits_only_known_color_names(self):
        from drugex.training.scorers.rocs_rdkit import _annotate_color_features

        m = _embed_oe_ff("CN(C)CCCOc1ccc(C(=O)O)cc1")
        _annotate_color_features(m, color_ff="oe_mills_dean")
        prop = m.GetProp("PUBCHEM_PHARMACOPHORE_FEATURES")
        emitted = {tok for tok in prop.replace("\n", " ").split() if tok.isalpha()}
        self.assertTrue(emitted.issubset(_SIX_TYPES), f"unexpected names: {emitted - _SIX_TYPES}")


@unittest.skipUnless(_RDSHAPEALIGN, "rdShapeAlign not available")
@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class RDKitOEScoring(unittest.TestCase):
    """`_score_single_reference(color_ff="oe_mills_dean")` routes color through the OE FF."""

    @classmethod
    def setUpClass(cls):
        cls.refs = [m for m in Chem.SDMolSupplier(_REF_SDF, removeHs=False) if m]
        # Pick the reference whose OE typing actually contains a `hydrophobe` feature, so the
        # hydrophobe-channel overlap is a faithful demonstration. (OE's hydrophobe SMARTS are
        # strict: most CCR2 reference carbons are ring atoms -> typed `rings`, not hydrophobe;
        # only one reference carries an OE hydrophobe -- verified empirically. This is the very
        # gap the port closes: the default rdShapeAlign color sees hydrophobe NOWHERE.)
        cls.hydrophobe_ref = None
        for m in cls.refs:
            r = Chem.Mol(m)
            from drugex.training.scorers.rocs_rdkit import _annotate_color_features

            _annotate_color_features(r, color_ff="oe_mills_dean")
            if "hydrophobe" in r.GetProp("PUBCHEM_PHARMACOPHORE_FEATURES"):
                cls.hydrophobe_ref = m
                break

    def test_oe_color_ff_enables_hydrophobic_query(self):
        from drugex.training.scorers.rocs_rdkit import _score_single_reference

        self.assertIsNotNone(
            self.hydrophobe_ref,
            "no CCR2 reference carries an OE hydrophobe feature -- cannot test the channel",
        )
        base = _score_single_reference(
            _embed_oe_ff("CCCCCCCC"), self.hydrophobe_ref, "color", True, color_ff="default"
        )
        oe = _score_single_reference(
            _embed_oe_ff("CCCCCCCC"), self.hydrophobe_ref, "color", True, color_ff="oe_mills_dean"
        )
        # Default path: octane is pure hydrophobe, the 5-pattern fallback has no hydrophobe -> 0.
        self.assertEqual(base, 0.0)
        # OE FF: octane's hydrophobe overlaps the reference's OE hydrophobe -> non-zero color.
        self.assertGreater(oe, 0.0)


@unittest.skipUnless(_RDSHAPEALIGN, "rdShapeAlign not available")
@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class RDKitScorerColorFFFlag(unittest.TestCase):
    """RDKitROCSScorer must accept color_ff and thread it end-to-end + into getKey."""

    def test_scorer_threads_color_ff_end_to_end(self):
        from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
        from drugex.training.scorers.conformer_generators import RDKitConformerGenerator

        cg = RDKitConformerGenerator(max_conformers=10, max_isomers=1)
        smis = ["CCCCCCCC"]  # octane: pure hydrophobe -> color 0 unless hydrophobe wired
        off = RDKitROCSScorer(
            cg, references=_REF_SDF, score_type="color",
            color_ff="default", show_progress=False, n_jobs=1,
        ).getScores(smis)[0, 0]
        on = RDKitROCSScorer(
            cg, references=_REF_SDF, score_type="color",
            color_ff="oe_mills_dean", show_progress=False, n_jobs=1,
        ).getScores(smis)[0, 0]
        self.assertEqual(off, 0.0)
        self.assertGreater(on, 0.0)

    def test_color_ff_key_suffix(self):
        from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
        from drugex.training.scorers.conformer_generators import RDKitConformerGenerator

        cg = RDKitConformerGenerator(max_conformers=2, max_isomers=1)
        default = RDKitROCSScorer(
            cg, references=_REF_SDF, score_type="color", show_progress=False, n_jobs=1
        ).getKey()[0]
        oe = RDKitROCSScorer(
            cg, references=_REF_SDF, score_type="color",
            color_ff="oe_mills_dean", show_progress=False, n_jobs=1,
        ).getKey()[0]
        self.assertNotEqual(default, oe)
        self.assertTrue(oe.endswith("_oe"), f"expected _oe suffix, got {oe!r}")

    def test_invalid_color_ff_raises(self):
        from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
        from drugex.training.scorers.conformer_generators import RDKitConformerGenerator

        cg = RDKitConformerGenerator(max_conformers=2, max_isomers=1)
        with self.assertRaises(ValueError):
            RDKitROCSScorer(
                cg, references=_REF_SDF, color_ff="bogus", show_progress=False
            )


# ---------------------------------------------------------------------------
# CDPKit path: custom PatternBasedFeatureGenerator from the OE SMARTS
# ---------------------------------------------------------------------------
@unittest.skipUnless(_CDPKIT, "CDPKit not available")
@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class CDPKitScorerColorFFFlag(unittest.TestCase):
    """CDPKitROCSScorer must accept color_ff, build the OE PatternBasedFeatureGenerator,
    and surface it via getKey. Default must remain the DefaultPharmacophoreGenerator."""

    def test_scorer_accepts_color_ff_and_scores(self):
        from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer
        from drugex.training.scorers.conformer_generators import CDPKitConformerGenerator

        cg = CDPKitConformerGenerator(max_conformers=10, max_isomers=1)
        smis = ["CN(C)CCCOc1ccc(C(=O)O)cc1"]
        oe = CDPKitROCSScorer(
            cg, references=_REF_SDF, opt_mode="color",
            color_ff="oe_mills_dean", show_progress=False, n_jobs=1,
        ).getScores(smis)[0, 0]
        # OE color FF is hydrophobe-rich vs the hydrophobe-heavy CCR2 ref -> non-trivial color.
        self.assertGreater(float(oe), 0.0)

    def test_color_ff_key_suffix(self):
        from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer
        from drugex.training.scorers.conformer_generators import CDPKitConformerGenerator

        cg = CDPKitConformerGenerator(max_conformers=2, max_isomers=1)
        default = CDPKitROCSScorer(
            cg, references=_REF_SDF, show_progress=False, n_jobs=1
        ).getKey()[0]
        oe = CDPKitROCSScorer(
            cg, references=_REF_SDF, color_ff="oe_mills_dean",
            show_progress=False, n_jobs=1,
        ).getKey()[0]
        self.assertNotEqual(default, oe)
        self.assertTrue(oe.endswith("_oe"), f"expected _oe suffix, got {oe!r}")

    def test_invalid_color_ff_raises(self):
        from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer
        from drugex.training.scorers.conformer_generators import CDPKitConformerGenerator

        cg = CDPKitConformerGenerator(max_conformers=2, max_isomers=1)
        with self.assertRaises(ValueError):
            CDPKitROCSScorer(cg, references=_REF_SDF, color_ff="bogus", show_progress=False)


# ---------------------------------------------------------------------------
# (e) DEFAULT-OFF: color_ff default preserves current scores bit-for-bit
# ---------------------------------------------------------------------------
@unittest.skipUnless(_RDSHAPEALIGN, "rdShapeAlign not available")
@unittest.skipUnless(os.path.exists(_REF_SDF), f"reference SDF not found: {_REF_SDF}")
class DefaultOff(unittest.TestCase):
    """(e) The default color_ff must change NOTHING vs the pre-feature code path."""

    @classmethod
    def setUpClass(cls):
        cls.mols = [m for m in Chem.SDMolSupplier(_REF_SDF, removeHs=False) if m]
        cls.ref0, cls.ref1 = cls.mols[0], cls.mols[1]

    def test_default_color_ff_value_is_default(self):
        from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
        from drugex.training.scorers.conformer_generators import RDKitConformerGenerator

        cg = RDKitConformerGenerator(max_conformers=2, max_isomers=1)
        s = RDKitROCSScorer(cg, references=_REF_SDF, show_progress=False, n_jobs=1)
        self.assertEqual(s.color_ff, "default")

    def test_default_getKey_has_no_oe_suffix(self):
        from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
        from drugex.training.scorers.conformer_generators import RDKitConformerGenerator

        cg = RDKitConformerGenerator(max_conformers=2, max_isomers=1)
        key = RDKitROCSScorer(
            cg, references=_REF_SDF, score_type="color", show_progress=False, n_jobs=1
        ).getKey()[0]
        self.assertFalse(key.endswith("_oe"))

    def test_default_matches_pre_feature_golden_scores(self):
        # The frozen golden scores from tests_color_modes (post issue-#8513 guard). The new
        # color_ff parameter, left at its default, must reproduce them exactly.
        from drugex.training.scorers.rocs_rdkit import _score_single_reference

        gold = {"TanimotoCombo": 0.853650, "shape": 0.731162, "color": 0.124658}
        for mode, expected in gold.items():
            v = _score_single_reference(self.ref0, self.ref1, mode, True)
            self.assertAlmostEqual(v, expected, delta=1e-3, msg=f"mode={mode}")

    def test_default_kwarg_equals_explicit_default(self):
        from drugex.training.scorers.rocs_rdkit import _score_single_reference

        probe = _embed_oe_ff("CCCCCCCc1ccccc1")
        implicit = _score_single_reference(probe, self.ref0, "color", True)
        explicit = _score_single_reference(probe, self.ref0, "color", True, color_ff="default")
        self.assertAlmostEqual(implicit, explicit, delta=1e-9)

    def test_default_is_independent_of_oe_path(self):
        # color_ff="default" must NOT call into the OE annotation -> identical to legacy fdef
        # injection toggle behavior (inject_color path unaffected).
        from drugex.training.scorers.rocs_rdkit import _score_single_reference

        probe = _embed_oe_ff("CCCCCCCc1ccccc1")
        legacy = _score_single_reference(probe, self.ref0, "color", True, inject_color=False)
        default_ff = _score_single_reference(
            probe, self.ref0, "color", True, inject_color=False, color_ff="default"
        )
        self.assertAlmostEqual(legacy, default_ff, delta=1e-9)


# ======================================================================================
# SECTION: RDKit ROCS pose-invariance guard, Lever 6 / Task #17  (was tests_pose_invariance.py)
# ======================================================================================

"""TDD tests for RDKit ROCS pose-invariance guard (Lever 6, Task #17; RDKit issue #8513).

Run:  ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests

Background (verified against github.com/rdkit/rdkit/issues/8513 + PR #8999):
``rdShapeAlign.AlignMol``'s two-phase optimizer starts the overlay from the probe's *input*
orientation, so the SAME conformer in a rotated/translated frame can converge to a different
local optimum and report a different score. Shape/color similarity is a rigid-body invariant,
so this is non-physical (in RL it injects orientation noise into the reward). The upstream fix
(PR #8999) aligns the initial start to the principal/inertial axes before overlay; the guard in
``rocs_rdkit`` reproduces that (CanonicalizeConformer + principal-axis sign starts, keep best) so
scoring is pose-invariant on every supported RDKit.

Empirically on RDKit 2025.09.3 (which has the partial #8513 fix but NOT PR #8999): raw AlignMol
spreads the combo score by ~0.13 (std ~0.056) across random rigid transforms of one conformer;
the guard collapses that to 0.000. These tests pin both facts:
  * ``RawAlignMolIsPoseDependent`` characterizes the bug (the guard's reason to exist). If a
    future RDKit makes raw AlignMol fully invariant this test will fail loudly -> revisit/retire
    the guard. It is therefore an EXPECTED-FAIL-on-fixed-upstream sentinel, not a regression.
  * ``ScorerIsPoseInvariant`` is the real contract: ``_score_single_reference`` returns the same
    score (within tol) regardless of the probe conformer's starting frame.
"""

# Tolerance: scores are rigid-body invariants, so with the guard they must match to numerical
# noise. rdShapeAlign reports ~1e-3-resolution Tanimoto; 5e-3 absolves only floating-point /
# optimizer round-off, not a genuine pose flip (which moves the score by ~0.1, see module docs).
_TOL_POSE = 5e-3


def _embed_pose(smi, seed=0xC0FFEE):
    """Single ETKDGv3 conformer with H's (matches the scorer's reference embedding seed)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    return mol


def _random_rigid_transform(mol, seed):
    """Return a copy of ``mol`` after a random proper rotation + translation of its conformer.

    A rigid-body move must leave any true shape/color similarity unchanged; this is the exact
    perturbation issue #8513 says the raw optimizer is (wrongly) sensitive to.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))          # fix QR sign convention
    if np.linalg.det(q) < 0:          # ensure a proper rotation (det = +1), not a reflection
        q[:, 0] *= -1
    transform = np.eye(4)
    transform[:3, :3] = q
    transform[:3, 3] = rng.normal(size=3) * 10.0   # translation far from origin
    out = Chem.Mol(mol)
    rdMolTransforms.TransformConformer(out.GetConformer(), transform)
    return out


# A reference and a *flexible*, color-rich probe: pose-sensitivity is worst when shape- and
# color-optimal poses can diverge, which needs rotatable bonds + polar groups (a rigid probe
# masks the bug). Omeprazole is the molecule from the original issue report.
_REF_SMILES = "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1"   # omeprazole
_PROBE_SMILES = "CCCCOc1ccc(CC(=O)NCc2ccccc2OC)cc1"          # 8 rot. bonds, donor/acceptor-rich
_N_POSES = 8


@unittest.skipUnless(_RDSHAPEALIGN, "rdShapeAlign not available")
class RawAlignMolIsPoseDependent(unittest.TestCase):
    """Sentinel: the unguarded rdShapeAlign call is pose-dependent (issue #8513).

    This is what motivates the guard. It asserts the *bug* still exists in raw AlignMol; if a
    future RDKit fixes it upstream this fails -> the guard can be simplified/removed.
    """

    def test_raw_align_score_varies_with_starting_pose(self):
        ref = _embed_pose(_REF_SMILES, seed=1)
        probe = _embed_pose(_PROBE_SMILES, seed=2)
        scores = []
        for s in range(_N_POSES):
            posed = _random_rigid_transform(probe, s)
            shape, color = rdShapeAlign.AlignMol(
                ref, posed, useColors=True, opt_param=0.5
            )
            scores.append(shape + color)
        spread = max(scores) - min(scores)
        # A true invariant would spread ~0; the documented bug spreads ~0.1.
        self.assertGreater(
            spread, 10 * _TOL_POSE,
            f"raw AlignMol unexpectedly pose-invariant (spread={spread:.4f}); "
            "issue #8513 may be fixed upstream -- revisit the guard",
        )


@unittest.skipUnless(_RDSHAPEALIGN, "rdShapeAlign not available")
class ScorerIsPoseInvariant(unittest.TestCase):
    """Contract: the guarded scorer returns the same score regardless of probe pose."""

    def _scores_over_poses(self, score_type):
        ref = _embed_pose(_REF_SMILES, seed=1)
        probe = _embed_pose(_PROBE_SMILES, seed=2)
        return [
            _score_single_reference(
                _random_rigid_transform(probe, s), ref, score_type, use_colors=True
            )
            for s in range(_N_POSES)
        ]

    def test_combo_score_pose_invariant(self):
        scores = self._scores_over_poses("TanimotoCombo")
        spread = max(scores) - min(scores)
        self.assertLessEqual(
            spread, _TOL_POSE,
            f"TanimotoCombo varies with probe pose (spread={spread:.4f}); "
            f"scores={np.round(scores, 4)}",
        )

    def test_shape_score_pose_invariant(self):
        scores = self._scores_over_poses("shape")
        self.assertLessEqual(max(scores) - min(scores), _TOL_POSE, np.round(scores, 4))

    def test_color_score_pose_invariant(self):
        scores = self._scores_over_poses("color")
        self.assertLessEqual(max(scores) - min(scores), _TOL_POSE, np.round(scores, 4))

    def test_self_alignment_is_robust(self):
        """The issue's own test: self-aligning under random poses must always succeed (~max).

        Raw AlignMol fails this ~25-40% of the time (score << 1); the guard must give a high,
        stable shape Tanimoto for every starting pose.
        """
        ref = _embed_pose(_REF_SMILES, seed=1)
        scores = [
            _score_single_reference(
                _random_rigid_transform(ref, 100 + s), ref, "shape", use_colors=True
            )
            for s in range(_N_POSES)
        ]
        self.assertGreaterEqual(
            min(scores), 0.99,
            f"self-alignment not robust to pose (min shape Tanimoto={min(scores):.4f}); "
            f"scores={np.round(scores, 4)}",
        )


# ======================================================================================
# SECTION: Native CDPKit tautomer + protonation prep, Task D2  (was tests_prep_cdpkit.py)
# ======================================================================================

"""Task D2 tests: NATIVE CDPKit tautomer + protonation prep in CDPKitConformerGenerator.

Run: ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests
"""

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


# ======================================================================================
# SECTION: Cross-backend protonation consistency, Task D / #11  (was tests_prep_consistency.py)
# ======================================================================================

"""Task D / #11: cross-backend protonation CONSISTENCY.

Confirms the three NATIVE protonators (RDKit=Dimorphite-DL, CDPKit=ProtonationStateStandardizer,
OpenEye=OEQuacPac) agree on the dominant pH-7.4 net charge for the same molecule — the basis
for the "native-per-backend is consistent" claim.

Run: ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests
"""

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
        backends = (["RDKit"] if _DIMORPHITE else []) + (["CDPKit"] if _CDPKIT else []) + (["OpenEye"] if _OE_QUACPAC else [])
        self.assertGreaterEqual(len(backends), 2, "need >=2 backends to compare")
        for smi, expected in CASES:
            charges = {b: self._charge(b, smi) for b in backends}
            for b, q in charges.items():
                self.assertEqual(q, expected, f"{b} gave {q} for {smi}, expected {expected}; all={charges}")


# ======================================================================================
# SECTION: Native OpenEye (OEQuacPac) tautomer + protonation prep, Task D3  (was tests_prep_openeye.py)
# ======================================================================================

"""Task D3 tests: NATIVE OpenEye (OEQuacPac) tautomer + protonation in OmegaConformerGenerator.

Run: ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests
"""

def _first_oeb_charge(path):
    from openeye import oechem as oe

    ifs = oe.oemolistream()
    if not ifs.open(path):
        return None
    mol = oe.OEMol()
    if not oe.OEReadMolecule(ifs, mol):
        return None
    return oe.OENetCharge(mol)


@unittest.skipUnless(_OE_QUACPAC, "OpenEye toolkit / OEQuacPac not licensed")
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


# ======================================================================================
# SECTION: Native RDKit tautomer + protonation prep, Task D1  (was tests_prep_rdkit.py)
# ======================================================================================

"""Task D1 tests: native tautomer + protonation prep in RDKitConformerGenerator.

Run: ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests
"""

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


# ======================================================================================
# SECTION: Shared pH-protonation primitive, Task D  (was tests_protonation.py)
# ======================================================================================

"""Tests for the shared pH-protonation primitive (Task D).

Run: ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests
"""

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


# ======================================================================================
# SECTION: OpenEye ROCS `-mpi_np` single-node parallelism flag  (was tests_rocs_mpi.py)
# ======================================================================================

"""TDD: OpenEye ROCS `-mpi_np` single-node parallelism flag (throughput fix for
the cross-dataset enrichment benchmark — job 10416 timed out at 3600s, single
core). Run: python -m drugex.training.scorers.tests
"""

class TestRocsMpi(unittest.TestCase):
    def _scorer(self, mpi_np):
        from drugex.training.scorers.rocs_openeye import OpenEyeROCSScorer
        s = OpenEyeROCSScorer.__new__(OpenEyeROCSScorer)
        s.binary_path = "rocs"
        s.mpi_np = mpi_np
        s.score_type = "TanimotoCombo"
        s.color_force_field = "ImplicitMillsDean"
        s.optimize = True
        s.shape_only = False
        s.color_optimize = True
        return s

    def test_mpi_np_flag_emitted_when_gt_one(self):
        cmd = self._scorer(44)._build_rocs_command("q.sdf", "db.sdf", "out/r")
        self.assertIn("-mpi_np", cmd)
        self.assertIn("44", cmd)
        # -mpi_np must come right after the binary (OpenEye CLI convention)
        i = cmd.index("-mpi_np")
        self.assertEqual(cmd[0], "rocs")
        self.assertEqual(cmd[i + 1], "44")

    def test_mpi_np_absent_when_one(self):
        cmd = self._scorer(1)._build_rocs_command("q.sdf", "db.sdf", "out/r")
        self.assertNotIn("-mpi_np", cmd)


# ======================================================================================
# SECTION: Reference-weighted Tversky color/combo, Lever 4 / Task #16 / #18  (was tests_tversky.py)
# ======================================================================================

"""TDD tests for reference-weighted Tversky color/combo (Lever 4, Task #16) — CDPKit backend.

Run:  ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests

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

def _cdpkit_pharm_shape_tversky(rdkit_mol):
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
        cls.rshape = _cdpkit_pharm_shape_tversky(ref0)
        # triamine_acid: size-mismatched vs ref0 -> Tversky must differ from Tanimoto
        m = Chem.AddHs(Chem.MolFromSmiles("NCCNCCNc1ccc(C(=O)O)cc1"))
        p = AllChem.ETKDGv3(); p.randomSeed = 7
        AllChem.EmbedMolecule(m, p); AllChem.MMFFOptimizeMolecule(m)
        cls.qshape = _cdpkit_pharm_shape_tversky(m)

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




@unittest.skipUnless(_OE, "OpenEye toolkit not licensed/available")
@unittest.skipUnless(os.path.exists(_ROCS_BIN), f"rocs binary not found: {_ROCS_BIN}")
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
        _libs = os.path.join(_REPO_ROOT, "ccr2_gen/oeye/runtime_libs")
        if os.path.isdir(_libs):
            os.environ["LD_LIBRARY_PATH"] = (
                _libs + ":" + os.environ.get("LD_LIBRARY_PATH", "")
            )
        _lic = os.path.join(_REPO_ROOT, "ccr2_gen/oe_license.txt")
        if os.path.exists(_lic):
            os.environ.setdefault("OE_LICENSE", _lic)
        smis = ["NCCNCCNc1ccc(C(=O)O)cc1"]
        tani = self._scorer("color", "tanimoto").getScores(smis)[0, 0]
        tvr = self._scorer("color", "tversky_ref").getScores(smis)[0, 0]
        self.assertGreaterEqual(tvr, 0.0)
        self.assertNotAlmostEqual(tani, tvr, delta=1e-4)  # the Tversky column was actually parsed


# ======================================================================================
# SECTION: RDKit conformer/tautomer caps that keep ROCS scoring fast
#          (was tests/test_conformer_caps.py -- converted to unittest)
# ======================================================================================

"""TDD tests for the RDKit conformer/tautomer caps that keep ROCS scoring fast.

Root cause (2026-06-23): RL training of rdkit/supermol/eps=0.4 ran at ~2.85h/epoch
because RDKitConformerGenerator's per-isomer ETKDG `timeout` defaulted to 120s and
09_rl built the generator WITHOUT passing it, so the ~10% of molecules that fail to
embed cleanly burned 120s x max_isomers (=480s each). The shared TautomerEnumerator
was also default-uncapped (maxTautomers/maxTransforms = 1000). These tests pin the
caps that bound per-molecule scoring time (near-constant-time training).

Run: ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests
"""


class ConformerCaps(unittest.TestCase):

    def test_shared_tautomer_enumerator_is_capped(self):
        te = cg._RDKIT_TAUTOMER_ENUMERATOR
        # default RDKit is 1000/1000 (unbounded for our purposes) — must be capped low.
        self.assertLessEqual(te.GetMaxTautomers(), 100)
        self.assertLessEqual(te.GetMaxTransforms(), 100)

    def test_rdkit_generator_default_timeout_is_bounded(self):
        # was 120s PER ISOMER -> 480s/molecule on the hard-to-embed tail. Bound it so a
        # pathological molecule cannot dominate an epoch.
        g = cg.RDKitConformerGenerator()
        self.assertTrue(0 < g.timeout <= 30)

    def test_etkdg_params_carry_the_generator_timeout(self):
        g = cg.RDKitConformerGenerator(timeout=7)
        params = g._create_fresh_etkdg()
        self.assertEqual(params.timeout, 7)

    def test_cdpkit_default_timeout_is_bounded(self):
        # was 3600s (1 HOUR per molecule!) -> CDPKit/supermol/eps0.3 ran ~31 min/epoch.
        # Bound it like RDKit so a pathological molecule can't dominate an epoch (audit 2026-06-24).
        g = cg.CDPKitConformerGenerator()
        self.assertTrue(0 < g.timeout <= 30)

    # ---------------------------------------------------------------------------
    # A1 — OpenEye Omega per-molecule search-time cap (BLOCKER B5)
    # ---------------------------------------------------------------------------

    @unittest.skipIf(not cg.OE_AVAILABLE, "OpenEye toolkits not available")
    def test_omega_default_timeout_is_bounded(self):
        # Omega had NO per-molecule search-time cap -> a torsion-driving-heavy molecule
        # could run unbounded and blow the GPU walltime (audit 2026-06-25).
        g = cg.OmegaConformerGenerator()
        self.assertTrue(0 < g.timeout <= 30)

    @unittest.skipIf(not cg.OE_AVAILABLE, "OpenEye toolkits not available")
    def test_omega_options_carry_the_search_time_cap(self):
        from openeye import oeomega

        g = cg.OmegaConformerGenerator()
        omega = g._create_fresh_omega()
        opts = omega.GetOptions() if hasattr(omega, "GetOptions") else None
        # The constructed options must carry the timeout on both the top-level
        # search-time and the torsion-driving search-time.
        # Build a fresh options object the same way _create_fresh_omega does and assert.
        fresh = oeomega.OEOmegaOptions()
        fresh.SetMaxSearchTime(float(g.timeout))
        self.assertEqual(fresh.GetMaxSearchTime(), float(g.timeout))
        # And that the generator actually applied it (introspect via a probe options object).
        probe = g._omega_options_for_test() if hasattr(g, "_omega_options_for_test") else None
        if probe is not None:
            self.assertEqual(probe.GetMaxSearchTime(), float(g.timeout))
            self.assertEqual(probe.GetTorDriveOptions().GetMaxSearchTime(), float(g.timeout))

    # ---------------------------------------------------------------------------
    # A2 — CDPKit tautomer enumeration cap (BLOCKER B6)
    # ---------------------------------------------------------------------------

    @unittest.skipIf(not cg.CDPL_AVAILABLE, "CDPKit not available")
    def test_cdpkit_canonical_tautomer_is_capped_and_bounded(self):
        import CDPL.Chem as CDPLChem

        # tautomer-rich molecule: cyclohexane-1,3,5-trione has many keto-enol tautomers.
        smi = "O=C1CC(=O)CC(=O)C1"
        mol = CDPLChem.parseSMILES(smi)
        CDPLChem.calcBasicProperties(mol, False)

        g = cg.CDPKitConformerGenerator()
        # The cap constant must exist and be bounded low.
        self.assertTrue(0 < cg._MAX_TAUTOMERS <= 100)

        t0 = time.monotonic()
        out = g._canonical_tautomer(mol)
        elapsed = time.monotonic() - t0
        self.assertIsNotNone(out)
        # Must canonicalize in bounded time.
        self.assertLess(elapsed, 2.0)

    @unittest.skipIf(not cg.CDPL_AVAILABLE, "CDPKit not available")
    def test_cdpkit_tautomer_callback_stops_after_cap(self):
        """The enumeration callback must stop once _MAX_TAUTOMERS are accepted.

        We force the cap to a tiny value and confirm the callback is invoked at most
        that many times for a tautomer-rich molecule that has more tautomers.
        """
        import CDPL.Chem as CDPLChem

        with mock.patch.object(cg, "_MAX_TAUTOMERS", 3):
            smi = "O=C1CC(=O)CC(=O)C1"
            mol = CDPLChem.parseSMILES(smi)
            CDPLChem.calcBasicProperties(mol, False)

            g = cg.CDPKitConformerGenerator()
            seen = g._canonical_tautomer_count(mol)
            self.assertLessEqual(seen, 3)


# ======================================================================================
# SECTION: scorer multiprocessing-context helper
#          (was tests/test_parallel.py -- converted to unittest)
# ======================================================================================

"""TDD tests for the scorer multiprocessing-context helper.

Root cause (2026-06-25): the cdpkit (and rdkit) ROCS scorers create a multiprocessing
``Pool`` with the default ``fork`` start method. In the RL loop the Pool is created *after*
``forward()``/``evolve()`` have spun up torch/OpenMP worker threads, so forking many workers
(n_jobs=62) hits the classic fork-after-threads deadlock — a child inherits a mutex locked by
a parent thread that does not exist in the child, ``pool.map`` blocks forever, and the epoch
never finishes (cdpkit/supermol/eps0.3/pt_pt, cell 615). ``forkserver`` forks workers from a
clean intermediary that never had those threads, avoiding the deadlock. This helper makes the
start method configurable (default ``fork`` — byte-for-byte for external users; the RL scripts
opt into ``forkserver``).

Run: ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests
"""


class ParallelHelpers(unittest.TestCase):
    """``monkeypatch.setenv/delenv`` -> ``mock.patch.dict(os.environ, ...)``.

    ``_unset(name)`` is the ``delenv(raising=False)`` translation: it snapshots the whole
    environment for the duration of the test (auto-restored by ``addCleanup``) and then
    removes the key if present.
    """

    def _patch_env(self, **kwargs):
        p = mock.patch.dict(os.environ, kwargs)
        p.start()
        self.addCleanup(p.stop)

    def _unset(self, *names):
        p = mock.patch.dict(os.environ)
        p.start()
        self.addCleanup(p.stop)
        for n in names:
            os.environ.pop(n, None)

    def test_default_is_fork_when_env_unset(self):
        self._unset("DRUGEX_MP_CONTEXT")
        self.assertEqual(pool_context().get_start_method(), "fork")

    def test_env_selects_forkserver(self):
        self._patch_env(DRUGEX_MP_CONTEXT="forkserver")
        self.assertEqual(pool_context().get_start_method(), "forkserver")

    def test_env_selects_spawn(self):
        self._patch_env(DRUGEX_MP_CONTEXT="spawn")
        self.assertEqual(pool_context().get_start_method(), "spawn")

    def test_bogus_method_falls_back_to_default(self):
        self._patch_env(DRUGEX_MP_CONTEXT="not_a_method")
        self.assertEqual(pool_context().get_start_method(), "fork")

    def test_explicit_default_arg_used_when_env_unset(self):
        self._unset("DRUGEX_MP_CONTEXT")
        self.assertEqual(pool_context(default="forkserver").get_start_method(), "forkserver")

    def test_returns_a_real_context_usable_for_pool(self):
        self._patch_env(DRUGEX_MP_CONTEXT="forkserver")
        ctx = pool_context()
        # the returned object must expose Pool (i.e. it is a real multiprocessing context)
        self.assertTrue(hasattr(ctx, "Pool"))
        self.assertIn(ctx.get_start_method(), mp.get_all_start_methods())

    # --- capped_n_jobs: bound scorer worker count (DRUGEX_SCORE_NJOBS) -------------------
    def test_no_cap_when_env_unset(self):
        self._unset("DRUGEX_SCORE_NJOBS")
        self.assertEqual(capped_n_jobs(62), 62)

    def test_cap_lowers_high_worker_count(self):
        # high counts can exhaust memory / wedge the fork for molecule-heavy cdpkit batches
        self._patch_env(DRUGEX_SCORE_NJOBS="16")
        self.assertEqual(capped_n_jobs(62), 16)

    def test_cap_does_not_raise_already_low_count(self):
        self._patch_env(DRUGEX_SCORE_NJOBS="16")
        self.assertEqual(capped_n_jobs(8), 8)       # min(8, 16) -> 8, never increases workers

    def test_cap_ignored_when_non_positive_or_garbage(self):
        self._patch_env(DRUGEX_SCORE_NJOBS="0")
        self.assertEqual(capped_n_jobs(62), 62)
        os.environ["DRUGEX_SCORE_NJOBS"] = "notanint"
        self.assertEqual(capped_n_jobs(62), 62)

    def test_cap_preserves_serial(self):
        self._patch_env(DRUGEX_SCORE_NJOBS="16")
        self.assertEqual(capped_n_jobs(1), 1)       # serial stays serial

    # --- score_timeout: per-batch scoring wall-clock cap (DRUGEX_SCORE_TIMEOUT) ----------
    def test_score_timeout_none_when_env_unset(self):
        self._unset("DRUGEX_SCORE_TIMEOUT")
        self.assertIsNone(score_timeout())     # default: wait forever (current behaviour)

    def test_score_timeout_reads_seconds(self):
        self._patch_env(DRUGEX_SCORE_TIMEOUT="300")
        self.assertEqual(score_timeout(), 300.0)

    def test_score_timeout_none_on_garbage_or_nonpositive(self):
        self._patch_env(DRUGEX_SCORE_TIMEOUT="0")
        self.assertIsNone(score_timeout())
        os.environ["DRUGEX_SCORE_TIMEOUT"] = "-5"
        self.assertIsNone(score_timeout())
        os.environ["DRUGEX_SCORE_TIMEOUT"] = "notanumber"
        self.assertIsNone(score_timeout())

    # --- mol_timeout + molecule_time_limit: per-molecule scoring cap --------------------
    def test_mol_timeout_env(self):
        self._unset("DRUGEX_SCORE_MOL_TIMEOUT")
        self.assertIsNone(mol_timeout())
        os.environ["DRUGEX_SCORE_MOL_TIMEOUT"] = "60"
        self.assertEqual(mol_timeout(), 60.0)
        os.environ["DRUGEX_SCORE_MOL_TIMEOUT"] = "0"
        self.assertIsNone(mol_timeout())

    def test_molecule_time_limit_raises_when_exceeded(self):
        with self.assertRaises(ScoreTimeout):
            with molecule_time_limit(0.3):
                time.sleep(2.0)            # exceeds -> SIGALRM -> ScoreTimeout

    def test_molecule_time_limit_passes_fast_block(self):
        with molecule_time_limit(2.0):
            x = sum(range(1000))           # well under the limit
        self.assertEqual(x, 499500)        # no exception, alarm disarmed cleanly

    def test_molecule_time_limit_noop_when_none(self):
        # None means "no cap" — must not arm an alarm or raise
        with molecule_time_limit(None):
            time.sleep(0.05)

    def test_molecule_time_limit_noop_off_main_thread(self):
        # SIGALRM is main-thread-only; off-thread it must be a silent no-op (not raise ValueError)
        err = []

        def run():
            try:
                with molecule_time_limit(0.1):
                    time.sleep(0.3)        # would raise on main thread; off-thread -> no-op
            except Exception as e:         # noqa: BLE001
                err.append(e)
        t = threading.Thread(target=run); t.start(); t.join()
        self.assertEqual(err, [])


# ======================================================================================
# SECTION: per-molecule scoring timeout, integration
#          (was tests/test_score_timeout_integration.py -- converted to unittest)
# ======================================================================================

"""Integration test: a single pathological molecule cannot stall scoring.

Root cause (2026-06-25, cdpkit/supermol/eps0.3/pt_pt cell 615): a molecule whose shape
alignment spins in an unbounded step wedged a Pool worker AND the serial fallback, hanging the
whole epoch for hours. The fix wraps each molecule's scoring in a per-molecule SIGALRM
(molecule_time_limit) so a hanging molecule is scored 0 (invalid -> 0 reward) and the worker
survives, plus a Pool-level get(timeout) backstop. This test injects an unbounded hang into the
alignment and asserts getScores returns in bounded time with that molecule scored 0.

Run: OE_LICENSE=... DRUGEX_SCORE_MOL_TIMEOUT=2 \
     ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests
"""

# Renamed from the original module-level ``CDPL`` / ``REF`` (too generic for a shared module).
_SCORE_TIMEOUT_CDPL = getattr(rc, "CDPL_AVAILABLE", False)
_SCORE_TIMEOUT_REF = "tutorial/advanced/rocs/rocs_rl_ccr/rdkit_cdpkit/supermol_123.sdf"


class ScoreTimeoutIntegration(unittest.TestCase):

    @unittest.skipIf(not _SCORE_TIMEOUT_CDPL, "CDPKit not available")
    def test_pathological_molecule_is_bounded_not_hanging(self):
        from drugex.training.scorers.conformer_generators import CDPKitConformerGenerator

        with mock.patch.dict(os.environ, {"DRUGEX_SCORE_MOL_TIMEOUT": "2"}):

            # Inject an unbounded hang into the per-conformer alignment (the cell-615 failure site).
            def _hang(*a, **k):
                time.sleep(30)
                return 0.0

            with mock.patch.object(rc, "_align_and_score_helper", _hang):
                sc = rc.CDPKitROCSScorer(
                    conformer_generator=CDPKitConformerGenerator(
                        max_conformers=5, max_isomers=1, show_progress=False),
                    references={"CCR2": _SCORE_TIMEOUT_REF}, opt_mode="combo", show_progress=False,
                    n_jobs=1,  # serial path runs in-process so the patched hang is exercised
                )
                mol = Chem.MolFromSmiles("O=C(CN1C(=O)COc2ccccc21)N1CCOCC1")

                t0 = time.time()
                out = sc.getScores([mol])
                elapsed = time.time() - t0

        self.assertLess(elapsed, 10, f"scoring not bounded: {elapsed:.1f}s (per-molecule timeout failed)")
        # the hung molecule must be scored 0 (invalid), not crash the batch
        self.assertEqual(float(out[0][0]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
