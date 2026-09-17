"""TDD tests for RDKit ROCS pose-invariance guard (Lever 6, Task #17; RDKit issue #8513).

Run:  ~/miniconda3/envs/drugex/bin/python -m drugex.training.scorers.tests_pose_invariance

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
import unittest

import numpy as np
from rdkit import Chem

try:
    from rdkit.Chem import rdShapeAlign, AllChem, rdMolTransforms  # noqa: F401
    _RDSHAPEALIGN = True
except ImportError:
    _RDSHAPEALIGN = False

from drugex.training.scorers.rocs_rdkit import _score_single_reference

# Tolerance: scores are rigid-body invariants, so with the guard they must match to numerical
# noise. rdShapeAlign reports ~1e-3-resolution Tanimoto; 5e-3 absolves only floating-point /
# optimizer round-off, not a genuine pose flip (which moves the score by ~0.1, see module docs).
_TOL = 5e-3


def _embed(smi, seed=0xC0FFEE):
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
        ref = _embed(_REF_SMILES, seed=1)
        probe = _embed(_PROBE_SMILES, seed=2)
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
            spread, 10 * _TOL,
            f"raw AlignMol unexpectedly pose-invariant (spread={spread:.4f}); "
            "issue #8513 may be fixed upstream -- revisit the guard",
        )


@unittest.skipUnless(_RDSHAPEALIGN, "rdShapeAlign not available")
class ScorerIsPoseInvariant(unittest.TestCase):
    """Contract: the guarded scorer returns the same score regardless of probe pose."""

    def _scores_over_poses(self, score_type):
        ref = _embed(_REF_SMILES, seed=1)
        probe = _embed(_PROBE_SMILES, seed=2)
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
            spread, _TOL,
            f"TanimotoCombo varies with probe pose (spread={spread:.4f}); "
            f"scores={np.round(scores, 4)}",
        )

    def test_shape_score_pose_invariant(self):
        scores = self._scores_over_poses("shape")
        self.assertLessEqual(max(scores) - min(scores), _TOL, np.round(scores, 4))

    def test_color_score_pose_invariant(self):
        scores = self._scores_over_poses("color")
        self.assertLessEqual(max(scores) - min(scores), _TOL, np.round(scores, 4))

    def test_self_alignment_is_robust(self):
        """The issue's own test: self-aligning under random poses must always succeed (~max).

        Raw AlignMol fails this ~25-40% of the time (score << 1); the guard must give a high,
        stable shape Tanimoto for every starting pose.
        """
        ref = _embed(_REF_SMILES, seed=1)
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


if __name__ == "__main__":
    unittest.main()
