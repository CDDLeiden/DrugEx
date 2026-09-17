"""RDKit-based ROCS scorer."""
import os
import tempfile
from collections import defaultdict
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import rdShapeAlign, AllChem, ChemicalFeatures, rdMolTransforms

from drugex.training.scorers.interfaces import ConformerGenerator, Scorer
from drugex.training.scorers.oe_color_smarts import OE_MILLS_DEAN_PATTERNS
from drugex.training.scorers.parallel import (
    pool_context, capped_n_jobs, mol_timeout, molecule_time_limit, ScoreTimeout,
)

_RDKIT_WORKER_SETTINGS: Dict[str, object] = {}
_DEFAULT_RDKIT_GROUP_NAME = "_default_group"
_VALID_SCORE_TYPES = frozenset({"shape", "color", "TanimotoCombo"})
_VALID_SCORE_VARIANTS = frozenset({"tanimoto", "tversky_ref", "tversky_fit"})
# color_ff selects which pharmacophore typing feeds rdShapeAlign's color channel via the
# PUBCHEM_PHARMACOPHORE_FEATURES property written by _annotate_color_features:
#   "default"       -> RDKit BaseFeatures.fdef typing (the existing inject_color path).
#   "fdef"          -> alias of "default" (explicit name for the BaseFeatures.fdef typing).
#   "oe_mills_dean" -> OpenEye ImplicitMillsDean SMARTS typing (Lever 1; oe_color_smarts.py).
# OPT-IN: "oe_mills_dean" is the only value that changes typing; the default path is unchanged
# (the annotation only runs at all when inject_color=True, exactly as before).
_VALID_COLOR_FFS = frozenset({"default", "fdef", "oe_mills_dean"})
# (alpha, beta) for the Tversky index; tversky_ref weights the REFERENCE (self_a).
_TVERSKY_AB = {"tversky_ref": (0.95, 0.05), "tversky_fit": (0.05, 0.95)}

# --- Color-feature annotation: the 6-type rdShapeAlign color path (incl. hydrophobe) ---
# Without an explicit PUBCHEM_PHARMACOPHORE_FEATURES property, rdShapeAlign perceives color
# with a 5-pattern fallback (donor/acceptor/rings/cation/anion) that has NO hydrophobe -- and
# the CCR2 references are hydrophobe-heavy, so RDKit is blind to their dominant color feature.
# Annotating from BaseFeatures.fdef activates the full 6-type scheme, matching OpenEye's
# ImplicitMillsDean color FF. See ccr2_gen/docs/color_validation/HOW_CLOSE_TO_OE.md §1b.
try:
    _COLOR_FACTORY = ChemicalFeatures.BuildFeatureFactory(
        os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    )
except Exception:  # pragma: no cover - BaseFeatures.fdef ships with RDKit
    _COLOR_FACTORY = None

# RDKit fdef family -> rdShapeAlign color type name (PUBCHEM_PHARMACOPHORE_FEATURES scheme).
# Per-atom "Hydrophobe" (not "LumpedHydrophobe") is used because LumpedHydrophobe does not
# fire on plain alkyl chains (e.g. octane); see Task #14 REFACTOR note on over-counting.
_FAMILY_TO_COLOR = {
    "Donor": "donor",
    "Acceptor": "acceptor",
    "PosIonizable": "cation",
    "NegIonizable": "anion",
    "Aromatic": "rings",
    "Hydrophobe": "hydrophobe",
}


def _color_lines_from_fdef(mol: Chem.Mol) -> List[str]:
    """PUBCHEM_PHARMACOPHORE_FEATURES body lines from RDKit BaseFeatures.fdef typing."""
    lines: List[str] = []
    for feat in _COLOR_FACTORY.GetFeaturesForMol(mol):
        color = _FAMILY_TO_COLOR.get(feat.GetFamily())
        if color is None:
            continue
        ids = [str(i + 1) for i in feat.GetAtomIds()]  # SDF atom indices are 1-based
        lines.append(f"{len(ids)} " + " ".join(ids) + f" {color}")
    return lines


def _color_lines_from_oe(mol: Chem.Mol) -> List[str]:
    """PUBCHEM_PHARMACOPHORE_FEATURES body lines from the OpenEye ImplicitMillsDean SMARTS.

    Each match of an OE color SMARTS becomes one feature line: ``<n> <atom ids...> <color>``
    with 1-based SDF atom ids (the same scheme the fdef path and rdShapeAlign's 6-type reader
    use). The OE port (oe_color_smarts.py) supplies precompiled patterns per color name.
    """
    lines: List[str] = []
    for color, patterns in OE_MILLS_DEAN_PATTERNS.items():
        seen: set = set()  # dedup identical atom sets a type may match via multiple patterns
        for patt in patterns:
            for match in mol.GetSubstructMatches(patt, uniquify=True):
                key = frozenset(match)
                if key in seen:
                    continue
                seen.add(key)
                ids = [str(i + 1) for i in match]
                lines.append(f"{len(ids)} " + " ".join(ids) + f" {color}")
    return lines


def _annotate_color_features(mol: Chem.Mol, color_ff: str = "default") -> Chem.Mol:
    """Write PUBCHEM_PHARMACOPHORE_FEATURES so rdShapeAlign's color uses the full 6-type
    scheme (donor/acceptor/cation/anion/rings/hydrophobe), matching OpenEye ImplicitMillsDean.

    ``color_ff`` selects the pharmacophore typing:
      "default" / "fdef" -> RDKit's BaseFeatures.fdef (the original behavior).
      "oe_mills_dean"    -> OpenEye ImplicitMillsDean SMARTS (Lever 1, oe_color_smarts.py),
                            which (unlike fdef) types saturated rings as ``rings`` and alkyl
                            chains as ``hydrophobe`` exactly as OpenEye does.

    Mutates ``mol`` in place and returns it. No-op when the (fdef) factory is unavailable or the
    molecule yields no mappable features (an empty feature block crashes rdShapeAlign).

    PRECONDITIONS before enabling ``inject_color`` in production (code-review #2/#3, Task #15):
    1. Scheme consistency — if ONE molecule is annotated (6-type scheme) and the other is not
       (e.g. a featureless molecule, or a reference SDF that already ships a foreign
       PUBCHEM_PHARMACOPHORE_FEATURES), rdShapeAlign perceives the un-annotated one with its
       5-pattern fallback whose integer type IDs differ -> color is computed on mismatched
       schemes. The shipped CCR2 references have no pre-existing property and all map features,
       so this is latent for them; annotate references ONCE in __init__ (overwriting any
       foreign property) before enabling injection on arbitrary reference sets.
    2. Mutation — this writes onto the passed object; in the sequential path that is the
       scorer's reference Mol (private when loaded from an SDF path, but caller-shared if a
       Chem.Mol was passed directly). Annotating refs once in __init__ also removes the
       per-call shared mutation.
    """
    if mol is None:
        return mol
    if color_ff == "oe_mills_dean":
        lines = _color_lines_from_oe(mol)
    else:  # "default" / "fdef": BaseFeatures.fdef typing
        if _COLOR_FACTORY is None:
            return mol
        lines = _color_lines_from_fdef(mol)
    if lines:  # guard: an empty feature block crashes rdShapeAlign
        mol.SetProp(
            "PUBCHEM_PHARMACOPHORE_FEATURES",
            f"{len(lines)}\n" + "\n".join(lines) + "\n",
        )
    return mol


def _tversky_from_tanimoto(
    tanimoto: float, self_a: float, self_b: float, alpha: float, beta: float
) -> float:
    """Tversky index on rdShapeAlign's NATIVE first-order Gaussian basis.

    rdShapeAlign reports only Tanimoto ``T = O_ab/(sov_a + sov_b - O_ab)`` but exposes the
    native self-overlaps (``ShapeInput.sov``/``.sof``). We recover the native cross-overlap
    ``O_ab = T*(self_a + self_b)/(1 + T)`` and form the Tversky index
    ``O_ab / (O_ab + alpha*(self_a - O_ab) + beta*(self_b - O_ab))``. With ``alpha=beta=1`` and
    a per-component Tanimoto ``T in [0,1]`` this returns ``T`` exactly -- the proof that the
    recovery is on the native basis. (The round-trip identity holds only for ``T<=1``; do NOT
    feed a TanimotoCombo in [0,2] here -- call per component. The final clamp would cap such a
    value at 1.0.) Result is clamped to [0, 1]: the first-order Gaussian overlap is an L2 inner
    product, so O_ab can exceed a self-overlap under extreme size mismatch -- for reference-
    weighted Tversky that legitimately means "reference fully covered" (saturates at 1.0).
    ``alpha`` weights ``self_a`` (pass the REFERENCE self-overlap as ``self_a``).
    """
    o_ab = tanimoto * (self_a + self_b) / (1.0 + tanimoto) if tanimoto > 0.0 else 0.0
    den = o_ab + alpha * (self_a - o_ab) + beta * (self_b - o_ab)
    if den <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, o_ab / den))


def _self_overlaps(mol: Chem.Mol, conf_id: int, opts) -> Tuple[float, float]:
    """Native (sov, sof) self-overlaps for a conformer (pose-independent)."""
    si = rdShapeAlign.PrepareConformer(mol, conf_id, opts)
    return si.sov, si.sof


# --- Pose-dependence guard for rdShapeAlign (RDKit issue #8513) -----------------------------
# rdShapeAlign.AlignMol's two-phase optimizer starts from the probe's *input* orientation, so
# the same conformer in a rotated/translated frame can converge to a different local optimum and
# report a different score (self-aligning omeprazole succeeds only ~75% of the time -- issue
# #8513, root-caused upstream in ncbi/pubchem-align3d#2). Shape/color similarity is a rigid-body
# invariant, so this is non-physical: in RL it injects orientation noise into the reward and lets
# two encodings of one molecule earn different scores.
#
# The fix landed in RDKit by aligning the initial start to the principal (inertial) axes before
# overlay (PR #8999, "Shape overlay initial start aligned to principal axes"; merged 2025-12-20,
# i.e. a post-2025.09 release). We reproduce that guard here so scoring is pose-invariant on every
# supported RDKit (verified: on 2025.09.3 the raw combo score spreads 0.13 / std 0.056 across
# random rigid transforms of one conformer; canonicalize+sign-starts collapses it to 0.000).
#
# Mechanism:
#   1. rdMolTransforms.CanonicalizeConformer puts each conformer in its inertial frame (center of
#      mass at origin, axes = principal moments) -> removes the arbitrary input orientation.
#   2. The inertial frame is sign-ambiguous (each principal axis is defined only up to +/-, the
#      residual non-determinism PR #8999 flagged), so we additionally try the 4 proper-rotation
#      (det=+1) 180-deg flips of the probe and keep the best overlay. This drives the cross-pose
#      spread to exactly 0 and self-alignment failures to 0/20.
# Reference molecules are canonicalized once up front; only the probe is flipped (the reference
# frame is fixed, so flipping the probe spans every relative axis-sign assignment).

# Proper rotations (determinant +1) that flip pairs of principal axes by 180 deg. The identity
# plus three two-axis flips cover all sign assignments reachable without an (improper) reflection.
_AXIS_SIGN_FLIPS: List[Tuple[int, int, int]] = [
    (1, 1, 1),
    (1, -1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
]


def _canonicalize_conformer_inplace(mol: Chem.Mol, conf_id: int) -> None:
    """Put one conformer of ``mol`` into its inertial frame in place (issue #8513 guard).

    No-op (swallowed) for degenerate geometries (e.g. <3 atoms, collinear) where the inertia
    tensor is singular; such molecules are pose-trivial anyway. ``conf_id`` of -1 selects the
    default conformer, matching rdShapeAlign's convention.
    """
    try:
        rdMolTransforms.CanonicalizeConformer(mol.GetConformer(conf_id))
    except (RuntimeError, ValueError):  # pragma: no cover - singular inertia tensor
        pass


def _flip_conformer(mol: Chem.Mol, conf_id: int, signs: Tuple[int, int, int]) -> Chem.Mol:
    """Return a copy of ``mol`` with conformer ``conf_id`` reflected by ``signs`` about the axes.

    ``signs`` must be a proper rotation (product of the three == +1) so chirality is preserved.
    """
    out = Chem.Mol(mol)
    transform = np.eye(4)
    transform[0, 0], transform[1, 1], transform[2, 2] = signs
    rdMolTransforms.TransformConformer(out.GetConformer(conf_id), transform)
    return out


def _score_single_reference(
    query_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    score_type: str,
    use_colors: bool,
    inject_color: bool = False,
    score_variant: str = "tanimoto",
    color_ff: str = "default",
) -> float:
    """Compute best alignment score between a query molecule and one reference.

    Uses rdShapeAlign.AlignMol which performs Gaussian shape overlay (same
    algorithm family as OpenEye ROCS and CDPKit GaussianShapeAlignment).

    ``color_ff`` selects the pharmacophore typing for the color channel (see
    ``_annotate_color_features``): "default"/"fdef" uses RDKit BaseFeatures.fdef,
    "oe_mills_dean" uses the OpenEye ImplicitMillsDean SMARTS port (Lever 1).
    Annotation is applied when ``inject_color`` is True OR ``color_ff`` is non-default
    (selecting an explicit FF implies injecting it); ``color_ff="default"`` with
    ``inject_color=False`` is the legacy no-op path.

    The alignment OPTIMIZATION objective is driven by ``score_type`` through
    AlignMol's ``opt_param`` (1.0 = shape-only, 0.5 = equal-weight shape+color
    ~ OpenEye ROCS ``-optchem`` TanimotoCombo, 0.0 = color-only). On flexible
    molecules this materially changes both the pose and the color score (an
    8-rotatable-bond probe vs a CCR2 reference gains ~0.10 ColorTanimoto under
    color optimization); it has little effect only on rigid references where
    the shape- and color-optimal poses coincide. (The earlier claim that
    opt_param has no effect held only at the rigid-reference / 200-conformer
    regime and is false for the flexible generated library.)
    """
    if query_mol is None or ref_mol is None:
        return 0.0
    if query_mol.GetNumConformers() == 0 or ref_mol.GetNumConformers() == 0:
        return 0.0

    # Activate the 6-type color path (incl. hydrophobe) by annotating both molecules.
    # Idempotent: each molecule is annotated once (refs/queries reused across calls), and
    # the per-conformer Chem.Mol copies below inherit the property. Annotation runs when the
    # caller asks for injection OR selects a non-default FF (an explicit FF means "inject it").
    if (inject_color or color_ff != "default") and use_colors:
        if not query_mol.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES"):
            _annotate_color_features(query_mol, color_ff=color_ff)
        if not ref_mol.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES"):
            _annotate_color_features(ref_mol, color_ff=color_ff)

    # Drive the alignment OPTIMIZATION objective from the requested score mode so that
    # color/combo modes co-optimize color rather than reading it off a shape-optimized
    # pose. rdShapeAlign.AlignMol opt_param: 1.0 = shape-only, 0.5 = equal-weight
    # shape+color (~ OpenEye ROCS -optchem TanimotoCombo), 0.0 = color-only.
    if score_type == "shape":
        opt_param = 1.0
    elif score_type == "color":
        opt_param = 0.0
    else:  # "TanimotoCombo" / combo
        opt_param = 0.5

    # For Tversky variants, precompute native (sov, sof) self-overlaps once per conformer; the
    # per-pose Tanimoto from AlignMol is converted to Tversky on the same native basis below.
    use_tversky = score_variant != "tanimoto"
    if use_tversky:
        alpha, beta = _TVERSKY_AB[score_variant]
        _opts = rdShapeAlign.ShapeInputOptions()
        _opts.useColors = use_colors
        ref_self = {rc.GetId(): _self_overlaps(ref_mol, rc.GetId(), _opts) for rc in ref_mol.GetConformers()}
        qry_self = {qc.GetId(): _self_overlaps(query_mol, qc.GetId(), _opts) for qc in query_mol.GetConformers()}

    # --- Pose-dependence guard (issue #8513): canonicalize both molecules into their inertial
    # frame, then start the probe from each principal-axis sign assignment and keep the best
    # overlay. Operate on COPIES so the shared/cross-process ref_mol and query_mol (and the
    # original-frame Tversky self-overlaps, which are pose-invariant) are untouched. The
    # reference frame is canonicalized ONCE per ref_conf (hoisted out of the query loop) and
    # reused so it stays fixed while the probe is flipped.
    ref_canon = Chem.Mol(ref_mol)
    for ref_conf in ref_mol.GetConformers():
        _canonicalize_conformer_inplace(ref_canon, ref_conf.GetId())

    best_score = 0.0
    for query_conf in query_mol.GetConformers():
        probe_canon = Chem.Mol(query_mol)
        _canonicalize_conformer_inplace(probe_canon, query_conf.GetId())
        for ref_conf in ref_mol.GetConformers():
            for signs in _AXIS_SIGN_FLIPS:
                probe_copy = _flip_conformer(probe_canon, query_conf.GetId(), signs)
                try:
                    result = rdShapeAlign.AlignMol(
                        ref_canon,
                        probe_copy,
                        refConfId=ref_conf.GetId(),
                        probeConfId=query_conf.GetId(),
                        useColors=use_colors,
                        opt_param=opt_param,
                    )
                except (RuntimeError, ValueError):
                    continue

                if not isinstance(result, (list, tuple)) or len(result) < 2:
                    continue

                shape_score, color_score = result[0], result[1]
                if use_tversky:
                    # ref_self/qry_self pass the REFERENCE self-overlap as self_a (alpha-weighted).
                    sov_r, sof_r = ref_self[ref_conf.GetId()]
                    sov_q, sof_q = qry_self[query_conf.GetId()]
                    shape_score = _tversky_from_tanimoto(shape_score, sov_r, sov_q, alpha, beta)
                    color_score = _tversky_from_tanimoto(color_score, sof_r, sof_q, alpha, beta)
                if score_type == "shape":
                    score = shape_score
                elif score_type == "color":
                    score = color_score
                else:
                    score = shape_score + color_score
                if score > best_score:
                    best_score = score
    return best_score


def _rdkit_worker_init(
    reference_mols: List[Chem.Mol],
    group_to_indices: List[List[int]],
    score_type: str,
    use_colors: bool,
    inject_color: bool = False,
    score_variant: str = "tanimoto",
    color_ff: str = "default",
) -> None:
    """Initializer to share immutable worker state."""
    global _RDKIT_WORKER_SETTINGS
    _RDKIT_WORKER_SETTINGS = {
        "reference_mols": reference_mols,
        "group_to_indices": group_to_indices,
        "score_type": score_type,
        "use_colors": use_colors,
        "inject_color": inject_color,
        "score_variant": score_variant,
        "color_ff": color_ff,
    }


def _score_molecule_rdkit_worker(args: Tuple[int, List[Chem.Mol]]) -> Tuple[int, List[float]]:
    """Score a molecule in a worker process."""
    mol_id, mol_conformers = args
    settings = _RDKIT_WORKER_SETTINGS
    reference_mols: List[Chem.Mol] = settings.get("reference_mols", [])
    group_to_indices: List[List[int]] = settings.get("group_to_indices", [])
    score_type: str = settings.get("score_type", "TanimotoCombo")
    use_colors: bool = settings.get("use_colors", True)
    inject_color: bool = settings.get("inject_color", False)
    score_variant: str = settings.get("score_variant", "tanimoto")
    color_ff: str = settings.get("color_ff", "default")
    num_groups = len(group_to_indices) if group_to_indices else (1 if reference_mols else 0)
    if not mol_conformers or num_groups == 0:
        return mol_id, [0.0] * num_groups

    group_scores = [0.0] * num_groups
    try:
        # Per-molecule wall-clock cap: a molecule whose alignment spins is scored 0 (invalid)
        # so it can't wedge the worker / serial fallback / epoch. No-op unless
        # DRUGEX_SCORE_MOL_TIMEOUT is set. (Same class of fix as cdpkit cell 615.)
        with molecule_time_limit(mol_timeout()):
            for conf_mol in mol_conformers:
                if conf_mol is None or conf_mol.GetNumConformers() == 0:
                    continue
                for group_idx, ref_indices in enumerate(group_to_indices):
                    for ref_idx in ref_indices:
                        ref_mol = reference_mols[ref_idx]
                        score = _score_single_reference(
                            conf_mol, ref_mol, score_type, use_colors, inject_color,
                            score_variant, color_ff,
                        )
                        if score > group_scores[group_idx]:
                            group_scores[group_idx] = score
    except ScoreTimeout:
        return mol_id, [0.0] * num_groups
    except (RuntimeError, ValueError, AttributeError, TypeError) as exc:
        import sys
        print(f"Warning: Error scoring molecule {mol_id}: {exc}", file=sys.stderr)
        return mol_id, [0.0] * num_groups

    return mol_id, group_scores


class RDKitROCSScorer(Scorer):
    """ROCS-style scoring using RDKit shape alignment.

    Features:
    - Accepts references as files or RDKit molecules with conformers.
    - Supports shape-only, color-only, and combo scoring modes.
    - Optional multiprocessing and verbose progress reporting.
    - Dict-based reference grouping for multi-target optimization.
    - SMILES deduplication for efficient batch processing.
    - Auto-conformer generation for 2D reference molecules.

    Attributes:
        conformer_generator: Generator used to build query conformers.
        reference_mols: Normalized list of reference molecules.
        group_to_indices: List mapping group indices to reference indices.
        group_names: List of reference group names.
        score_type: Requested score mode (`shape`, `color`, or `TanimotoCombo`).
        use_colors: Whether to include pharmacophore colors in alignment.
        show_progress: Enables logging for long runs.
        n_jobs: Requested worker count (-1 maps to available CPUs).
        _single_reference: True when operating in single-reference mode.
    """

    def __init__(
        self,
        conformer_generator: ConformerGenerator,
        references: Union[
            str,
            List[str],
            Dict[str, List[str]],
            Chem.Mol,
            List[Chem.Mol],
            Dict[str, List[Chem.Mol]],
        ],
        score_type: str = "TanimotoCombo",
        use_colors: bool = True,
        show_progress: bool = True,
        n_jobs: int = -1,
        inject_color: bool = False,
        score_variant: str = "tanimoto",
        color_ff: str = "default",
    ):
        """Initialize the RDKit ROCS scorer.

        Args:
            conformer_generator: Conformer generator used for query molecules.
            references: Reference source as SDF path(s), RDKit molecule(s),
                or a dict mapping group names to lists of references.
            score_type: Score variant (`TanimotoCombo`, `shape`, `color`).
            use_colors: Whether to include pharmacophore colors in alignments.
            show_progress: Enables stdout progress updates when True.
            n_jobs: Number of worker processes (-1 uses all available CPUs).
            inject_color: When True, annotate query + reference molecules with the
                6-type PUBCHEM_PHARMACOPHORE_FEATURES scheme (incl. hydrophobe) so
                rdShapeAlign's color matches OpenEye ImplicitMillsDean rather than its
                5-pattern fallback. Off by default (changes color scores; recalibrate
                thresholds before enabling in production).
            color_ff: Pharmacophore typing for the injected color features:
                "default"/"fdef" = RDKit BaseFeatures.fdef (legacy), "oe_mills_dean" =
                OpenEye ImplicitMillsDean SMARTS port (Lever 1). Selecting "oe_mills_dean"
                implies color injection with OE typing. OPT-IN: the default leaves color
                scoring bit-for-bit unchanged. Recalibrate thresholds before making it
                the default (the OE-ground-truth Spearman GATE is a separate compute step).

        Raises:
            TypeError: If reference inputs use unsupported types.
            FileNotFoundError: If a reference path is missing.
            ValueError: If reference molecules lack conformers or cannot load.
        """
        super().__init__()

        if score_type not in _VALID_SCORE_TYPES:
            raise ValueError(
                f"score_type must be one of {sorted(_VALID_SCORE_TYPES)}, got {score_type!r}"
            )
        if score_variant not in _VALID_SCORE_VARIANTS:
            raise ValueError(
                f"score_variant must be one of {sorted(_VALID_SCORE_VARIANTS)}, "
                f"got {score_variant!r}"
            )
        if color_ff not in _VALID_COLOR_FFS:
            raise ValueError(
                f"color_ff must be one of {sorted(_VALID_COLOR_FFS)}, got {color_ff!r}"
            )

        self.conformer_generator = conformer_generator
        self.score_type = score_type
        self.use_colors = use_colors
        self.inject_color = inject_color
        self.score_variant = score_variant
        self.color_ff = color_ff
        self.show_progress = show_progress
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.n_jobs = capped_n_jobs(self.n_jobs)  # DRUGEX_SCORE_NJOBS bound (cell-615 mem hang)

        self.group_definitions = self._prepare_reference_groups(references)
        self.group_names = [name for name, _ in self.group_definitions]
        self.reference_mols, self.group_to_indices = self._flatten_groups(self.group_definitions)
        self.reference_mols = [self._ensure_reference_conformers(m) for m in self.reference_mols]
        self._validate_references()
        self._single_reference = len(self.reference_mols) == 1

    def _prepare_reference_groups(
        self,
        references: Union[
            str,
            List[str],
            Dict[str, List[str]],
            Chem.Mol,
            List[Chem.Mol],
            Dict[str, List[Chem.Mol]],
        ],
    ) -> List[Tuple[str, List[Chem.Mol]]]:
        groups: List[Tuple[str, List[Chem.Mol]]] = []

        if isinstance(references, dict):
            for name, refs in references.items():
                ref_mols = self._normalize_reference_collection(refs)
                groups.append((str(name), ref_mols))
        else:
            ref_mols = self._normalize_reference_collection(references)
            groups.append((_DEFAULT_RDKIT_GROUP_NAME, ref_mols))

        if not groups:
            raise ValueError("At least one reference group must be provided")
        return groups

    def _normalize_reference_collection(
        self,
        refs: Union[str, List[str], Chem.Mol, List[Chem.Mol]],
    ) -> List[Chem.Mol]:
        if isinstance(refs, (str, Chem.Mol)):
            refs = [refs]
        if not isinstance(refs, list):
            raise TypeError(
                "references must be str, List[str], Chem.Mol, List[Chem.Mol], or dict thereof"
            )

        normalized: List[Chem.Mol] = []
        for item in refs:
            if isinstance(item, str):
                normalized.extend(self._load_molecules_from_file(item))
            elif isinstance(item, Chem.Mol):
                normalized.append(item)
            else:
                raise TypeError(
                    "Reference entries must be file paths or RDKit molecules"
                )

        if not normalized:
            raise ValueError("Reference group cannot be empty")
        return normalized

    def _flatten_groups(
        self, groups: List[Tuple[str, List[Chem.Mol]]]
    ) -> Tuple[List[Chem.Mol], List[List[int]]]:
        reference_mols: List[Chem.Mol] = []
        group_to_indices: List[List[int]] = []

        for _, refs in groups:
            indices: List[int] = []
            for ref in refs:
                indices.append(len(reference_mols))
                reference_mols.append(ref)
            group_to_indices.append(indices)

        return reference_mols, group_to_indices

    def _load_molecules_from_file(self, path: str) -> List[Chem.Mol]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Reference file not found: {path}")

        try:
            suppl = Chem.SDMolSupplier(path, removeHs=False)
            if not suppl:
                raise ValueError(f"Could not open SDF file: {path}")
            mols = [m for m in suppl if m is not None and m.GetNumAtoms() > 0]
            if not mols:
                raise ValueError(f"No molecules found in file: {path}")
            return mols
        except Exception as exc:
            if self.show_progress:
                print(f"Warning: failed to load {path}: {exc}")
            raise ValueError(f"Failed to load molecules from {path}: {exc}") from exc

    def _ensure_reference_conformers(self, mol: Chem.Mol) -> Chem.Mol:
        if mol is None:
            return mol
        if mol.GetNumConformers() > 0:
            return mol
        try:
            m = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xC0FFEE
            AllChem.EmbedMolecule(m, params=params)
            return m if m.GetNumConformers() > 0 else mol
        except Exception as exc:
            if self.show_progress:
                print(f"Warning: embedding reference failed: {exc}")
            return mol

    def _validate_references(self):
        if not self.reference_mols:
            raise ValueError("At least one reference molecule is required")

        index_map: Dict[int, int] = {}
        valid_refs: List[Chem.Mol] = []
        for idx, ref_mol in enumerate(self.reference_mols):
            if ref_mol is None or ref_mol.GetNumConformers() == 0:
                if self.show_progress:
                    print(
                        f"Warning: reference molecule at index {idx} is invalid or lacks conformers"
                    )
                continue
            index_map[idx] = len(valid_refs)
            valid_refs.append(ref_mol)

        if not valid_refs:
            raise ValueError("No valid reference molecules with conformers available")

        new_group_to_indices: List[List[int]] = []
        for name, indices in zip(self.group_names, self.group_to_indices):
            mapped = [index_map[i] for i in indices if i in index_map]
            if not mapped:
                raise ValueError(
                    f"Reference group '{name}' has no valid molecules with conformers"
                )
            new_group_to_indices.append(mapped)

        self.reference_mols = valid_refs
        self.group_to_indices = new_group_to_indices

    def _key_suffix(self) -> str:
        """Distinguish scorers by non-default flags so keys don't collide (defaults -> '')."""
        suffix = ""
        if self.score_variant != "tanimoto":
            suffix += f"_{self.score_variant}"
        if self.inject_color:
            suffix += "_inject"
        if self.color_ff == "oe_mills_dean":
            suffix += "_oe"
        return suffix

    def getKey(self) -> List[str]:
        sfx = self._key_suffix()
        if (
            len(self.group_names) == 1
            and self.group_names[0] == _DEFAULT_RDKIT_GROUP_NAME
        ):
            prefix = "RDKit_Supermol" if self._single_reference else "RDKit_Aggregate"
            refs = len(self.reference_mols)
            if prefix == "RDKit_Aggregate":
                return [f"{prefix}_{refs}refs_{self.score_type}{sfx}"]
            return [f"{prefix}_{self.score_type}{sfx}"]
        return [f"RDKit_{name}{sfx}" for name in self.group_names]

    @staticmethod
    def _deduplicate_smiles(
        smiles_list: List[Union[str, None]]
    ) -> Tuple[List[str], Dict[int, List[int]]]:
        """Group identical SMILES to avoid redundant conformer generation."""
        unique_smiles: List[str] = []
        unique_lookup: Dict[str, int] = {}
        unique_to_original: Dict[int, List[int]] = defaultdict(list)

        for idx, smi in enumerate(smiles_list):
            if smi is None:
                continue
            unique_idx = unique_lookup.get(smi)
            if unique_idx is None:
                unique_idx = len(unique_smiles)
                unique_smiles.append(smi)
                unique_lookup[smi] = unique_idx
            unique_to_original[unique_idx].append(idx)

        return unique_smiles, unique_to_original

    def _convert_to_smiles(self, mols) -> List[Union[str, None]]:
        smiles_list: List[Union[str, None]] = []
        for mol in mols:
            if mol is None:
                smiles_list.append(None)
            elif isinstance(mol, str):
                smiles_list.append(mol)
            else:
                try:
                    smiles_list.append(Chem.MolToSmiles(mol))
                except Exception:
                    smiles_list.append(None)
        return smiles_list

    def _score_sequential(
        self,
        unique_count: int,
        conformers_by_mol: Dict[int, List[Chem.Mol]],
        num_groups: int,
    ) -> np.ndarray:
        """Score molecules sequentially without multiprocessing.

        Args:
            unique_count: Number of unique molecules to score.
            conformers_by_mol: Dictionary mapping molecule IDs to their conformer lists.
            num_groups: Number of reference groups.

        Returns:
            Array of shape (unique_count, num_groups) containing scores.
        """
        scores_unique = np.zeros((unique_count, num_groups))

        for mol_id in range(unique_count):
            mol_conformers = conformers_by_mol.get(mol_id, [])
            if not mol_conformers:
                continue
            group_scores = np.zeros(num_groups)
            for conf_mol in mol_conformers:
                for group_idx, ref_indices in enumerate(self.group_to_indices):
                    for ref_idx in ref_indices:
                        ref_mol = self.reference_mols[ref_idx]
                        score = _score_single_reference(
                            conf_mol, ref_mol, self.score_type, self.use_colors,
                            self.inject_color, self.score_variant, self.color_ff,
                        )
                        if score > group_scores[group_idx]:
                            group_scores[group_idx] = score
            scores_unique[mol_id] = group_scores
            if self.show_progress and (mol_id + 1) % 100 == 0:
                print(f"  Scored {mol_id + 1}/{unique_count} unique molecules")

        return scores_unique

    def getScores(
        self, mols: List[Chem.Mol], frags: Optional[List[Chem.Mol]] = None
    ) -> np.ndarray:
        num_groups = len(self.group_to_indices)
        if num_groups == 0:
            raise ValueError("No reference groups configured")

        if not mols:
            return np.zeros((0, num_groups))

        num_mols = len(mols)
        scores = np.zeros((num_mols, num_groups))

        if self.show_progress:
            print(f"Scoring {num_mols} molecules with {self.getKey()}...")

        smiles_list = self._convert_to_smiles(mols)
        unique_smiles, unique_to_original = self._deduplicate_smiles(smiles_list)

        if not unique_smiles:
            return scores

        # ignore_cleanup_errors: on networked scratch (NFS/Lustre) a lingering file handle at
        # exit becomes a silly-rename (.nfsXXXX) so rmtree raises ENOTEMPTY *after* scores are
        # computed -> the result would be silently lost. Swallow the cleanup error (the dir is
        # reaped by the job's TMPDIR cleanup).
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            conf_file = self.conformer_generator.genConformers(unique_smiles, tmpdir)
            if not os.path.exists(conf_file):
                if self.show_progress:
                    print("Warning: conformer generation failed")
                return scores

            conformers_by_mol = defaultdict(list)
            try:
                suppl = Chem.SDMolSupplier(conf_file, removeHs=False)
                if suppl is None:
                    if self.show_progress:
                        print(f"Warning: Could not open SDF file: {conf_file}")
                    return scores
            except Exception as exc:
                if self.show_progress:
                    print(f"Warning: Failed to open conformer file {conf_file}: {exc}")
                return scores

            for conf_mol in suppl:
                if conf_mol is None:
                    continue
                try:
                    name = conf_mol.GetProp("_Name")
                    parts = name.split("+")[0].split("_")
                    if len(parts) < 2:
                        if self.show_progress:
                            print(f"Warning: Malformed conformer name: {name}")
                        continue
                    mol_id = int(parts[1])
                    conformers_by_mol[mol_id].append(conf_mol)
                except (KeyError, ValueError, IndexError) as exc:
                    if self.show_progress:
                        print(f"Warning: Could not parse conformer name: {exc}")
                    continue

            unique_count = len(unique_smiles)

            # Initialize score array for all code paths
            scores_unique = np.zeros((unique_count, num_groups))

            if self.n_jobs == 1:
                scores_unique = self._score_sequential(
                    unique_count, conformers_by_mol, num_groups
                )
            else:
                worker_args = [
                    (mol_id, conformers_by_mol.get(mol_id, []))
                    for mol_id in range(unique_count)
                ]
                effective_jobs = max(1, self.n_jobs)
                chunksize = max(1, unique_count // (effective_jobs * 4))

                try:
                    # forkserver (via DRUGEX_MP_CONTEXT) avoids the fork-after-threads
                    # deadlock that hangs a fork-Pool created after torch/OpenMP threads
                    # exist (see parallel.pool_context + cdpkit cell 615).
                    with pool_context().Pool(
                        self.n_jobs,
                        initializer=_rdkit_worker_init,
                        initargs=(
                            self.reference_mols,
                            self.group_to_indices,
                            self.score_type,
                            self.use_colors,
                            self.inject_color,
                            self.score_variant,
                            self.color_ff,
                        ),
                    ) as pool:
                        if self.show_progress:
                            try:
                                from tqdm import tqdm

                                results = list(
                                    tqdm(
                                        pool.imap(
                                            _score_molecule_rdkit_worker,
                                            worker_args,
                                            chunksize=chunksize,
                                        ),
                                        total=len(worker_args),
                                        desc="Scoring unique molecules",
                                    )
                                )
                            except ImportError:
                                results = pool.map(
                                    _score_molecule_rdkit_worker,
                                    worker_args,
                                    chunksize=chunksize,
                                )
                                print(
                                    f"  Scored {len(worker_args)} unique molecules "
                                    "(parallel)"
                                )
                        else:
                            results = pool.map(
                                _score_molecule_rdkit_worker,
                                worker_args,
                                chunksize=chunksize,
                            )
                    for mol_id, group_scores in results:
                        if 0 <= mol_id < unique_count and len(group_scores) == num_groups:
                            scores_unique[mol_id] = np.asarray(group_scores)
                except Exception as exc:
                    if self.show_progress:
                        print(
                            f"Warning: parallel processing failed ({exc}), "
                            "switching to sequential mode"
                        )
                    scores_unique = self._score_sequential(
                        unique_count, conformers_by_mol, num_groups
                    )

        for unique_id, original_indices in unique_to_original.items():
            if unique_id >= scores_unique.shape[0]:
                continue
            for original_idx in original_indices:
                scores[original_idx] = scores_unique[unique_id]

        if self.show_progress:
            non_zero = np.count_nonzero(scores)
            avg_score = scores.mean()
            max_score = scores.max() if scores.size > 0 else 0.0
            print(
                f"Scoring complete. Average score: {avg_score:.3f}, "
                f"Max score: {max_score:.3f}, "
                f"Molecules with score > 0: {non_zero}/{scores.shape[0]}"
            )

        return scores
