"""TDD tests for RDKit color hydrophobe injection (Lever 2, Task #14).

Run:  ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests_hydrophobe_color

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
import os
import unittest

from rdkit import Chem

try:
    from rdkit.Chem import rdShapeAlign, AllChem  # noqa: F401
    _RDSHAPEALIGN = True
except ImportError:
    _RDSHAPEALIGN = False

# RED: this symbol does not exist yet -> module import fails -> all tests error (expected).
from drugex.training.scorers.rocs_rdkit import _annotate_color_features, _score_single_reference

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_REF_SDF = os.path.join(
    _REPO_ROOT, "tutorial/advanced/rocs/rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf"
)


def _embed(smi):
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
        self.assertEqual(self._self_color(_embed("CCCCCCCC")), 0.0)

    def test_octane_gains_color_after_injection(self):
        """The core behavior: after annotation the hydrophobe channel fires -> color > 0."""
        m = _embed("CCCCCCCC")
        _annotate_color_features(m)
        self.assertGreater(self._self_color(m), 0.0)

    def test_injection_preserves_polar_color(self):
        """Annotation must not destroy existing donor/acceptor color."""
        m = _embed("OCCO")  # ethylene glycol: donors + acceptors
        _annotate_color_features(m)
        self.assertGreater(self._self_color(m), 0.0)

    def test_annotation_sets_pubchem_property(self):
        """The function writes the 6-type SD property rdShapeAlign reads."""
        m = _embed("CCCCCCCC")
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
        base = _score_single_reference(_embed("CCCCCCCC"), self.ref0, "color", True, inject_color=False)
        inj = _score_single_reference(_embed("CCCCCCCC"), self.ref0, "color", True, inject_color=True)
        self.assertEqual(base, 0.0)
        self.assertGreater(inj, 0.0)

    def test_inject_color_default_off_preserves_behavior(self):
        """Default (no inject_color kwarg) must equal inject_color=False (no silent change)."""
        probe = _embed("CCCCCCCc1ccccc1")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
