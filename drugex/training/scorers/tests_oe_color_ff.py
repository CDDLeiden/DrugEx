"""TDD tests for the OpenEye ImplicitMillsDean color force field port (Lever 1).

Run (with OE_LICENSE + LD_LIBRARY_PATH from ccr2_gen/oeye/runtime_libs):
  ~/miniconda3/envs/drugex/bin/python -m pytest \
      drugex/training/scorers/tests_oe_color_ff.py -q

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
import os
import unittest

from rdkit import Chem

try:
    from rdkit.Chem import rdShapeAlign, AllChem  # noqa: F401
    _RDSHAPEALIGN = True
except ImportError:
    _RDSHAPEALIGN = False

try:
    import CDPL.Pharm as _CDPLPharm  # noqa: F401
    _CDPKIT = True
except ImportError:
    _CDPKIT = False

# RED: this module does not exist yet -> import fails -> every test errors (expected).
from drugex.training.scorers.oe_color_smarts import (
    OE_MILLS_DEAN_IMPLICIT,
    OE_TO_CDPKIT_TYPE,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_REF_SDF = os.path.join(
    _REPO_ROOT, "tutorial/advanced/rocs/rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf"
)

_SIX_TYPES = {"donor", "acceptor", "cation", "anion", "rings", "hydrophobe"}


def _embed(smi, seed=7):
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

        m = _embed("CCCCCCCC")
        _annotate_color_features(m, color_ff="oe_mills_dean")
        self.assertTrue(m.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES"))
        self.assertIn("hydrophobe", m.GetProp("PUBCHEM_PHARMACOPHORE_FEATURES"))

    def test_oe_annotation_emits_saturated_ring_as_rings(self):
        # cyclohexane: OE `rings` fires (fdef Aromatic would NOT) -> proves OE path is used.
        from drugex.training.scorers.rocs_rdkit import _annotate_color_features

        m = _embed("C1CCCCC1")
        _annotate_color_features(m, color_ff="oe_mills_dean")
        self.assertIn("rings", m.GetProp("PUBCHEM_PHARMACOPHORE_FEATURES"))

    def test_oe_annotation_emits_only_known_color_names(self):
        from drugex.training.scorers.rocs_rdkit import _annotate_color_features

        m = _embed("CN(C)CCCOc1ccc(C(=O)O)cc1")
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
            _embed("CCCCCCCC"), self.hydrophobe_ref, "color", True, color_ff="default"
        )
        oe = _score_single_reference(
            _embed("CCCCCCCC"), self.hydrophobe_ref, "color", True, color_ff="oe_mills_dean"
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

        probe = _embed("CCCCCCCc1ccccc1")
        implicit = _score_single_reference(probe, self.ref0, "color", True)
        explicit = _score_single_reference(probe, self.ref0, "color", True, color_ff="default")
        self.assertAlmostEqual(implicit, explicit, delta=1e-9)

    def test_default_is_independent_of_oe_path(self):
        # color_ff="default" must NOT call into the OE annotation -> identical to legacy fdef
        # injection toggle behavior (inject_color path unaffected).
        from drugex.training.scorers.rocs_rdkit import _score_single_reference

        probe = _embed("CCCCCCCc1ccccc1")
        legacy = _score_single_reference(probe, self.ref0, "color", True, inject_color=False)
        default_ff = _score_single_reference(
            probe, self.ref0, "color", True, inject_color=False, color_ff="default"
        )
        self.assertAlmostEqual(legacy, default_ff, delta=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
