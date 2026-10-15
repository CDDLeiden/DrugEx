"""CDPKit-based ROCS scorer implementation."""

import os
import tempfile
from collections import defaultdict
from multiprocessing import cpu_count, TimeoutError as MPTimeoutError
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem

try:
    import CDPL.Chem as CDPLChem
    import CDPL.Shape as CDPLShape
    import CDPL.Pharm as CDPLPharm

    CDPL_AVAILABLE = True
except ImportError:
    CDPL_AVAILABLE = False
    CDPLChem = None
    CDPLShape = None
    CDPLPharm = None

from drugex.training.scorers.interfaces import ConformerGenerator, Scorer
from drugex.training.scorers.oe_color_smarts import build_cdpkit_oe_pharm_generator
from drugex.training.scorers.parallel import (
    pool_context, capped_n_jobs, score_timeout, mol_timeout, molecule_time_limit, ScoreTimeout,
)

MAX_OPTIMIZATION_ITERATIONS = 20
OPTIMIZATION_STOP_GRADIENT = 1.0

_DEFAULT_CDPKIT_GROUP_NAME = "_default_group"

_VALID_OPT_MODES = frozenset({"shape", "combo", "color"})
_VALID_SCORE_VARIANTS = frozenset({"tanimoto", "tversky_ref", "tversky_fit"})
# color_ff selects the pharmacophore generator feeding the colored Gaussian shape:
#   "default" -> CDPKit DefaultPharmacophoreGenerator (legacy, unchanged).
#   "oe_mills_dean" -> OpenEye ImplicitMillsDean SMARTS via a custom PatternBasedFeature
#                      generator (Lever 1; oe_color_smarts.build_cdpkit_oe_pharm_generator).
# OPT-IN: "oe_mills_dean" is the only value that changes color typing.
_VALID_COLOR_FFS = frozenset({"default", "oe_mills_dean"})

# Per-process cache of the (unpicklable) OE pharmacophore generator. Built lazily the first
# time a worker/loader needs it; CDPKit C++ generator objects can't cross the Pool boundary,
# so we ship the color_ff string and reconstruct the generator inside each process.
_OE_PHARM_GEN_CACHE = None


def _get_oe_pharm_generator():
    """Return this process's OE ImplicitMillsDean pharmacophore generator (built once)."""
    global _OE_PHARM_GEN_CACHE
    if _OE_PHARM_GEN_CACHE is None:
        _OE_PHARM_GEN_CACHE = build_cdpkit_oe_pharm_generator()
    return _OE_PHARM_GEN_CACHE

# Built once at import (not per scoring call). Maps (score_variant x opt_mode) -> the CDPKit
# native score function. Empty when CDPKit is unavailable (the scorer raises ImportError first).
if CDPL_AVAILABLE:
    _CDPKIT_SCORE_FUNCS: Dict[str, Dict[str, object]] = {
        "tanimoto": {
            "shape": CDPLShape.calcShapeTanimotoScore,
            "color": CDPLShape.calcColorTanimotoScore,
            "combo": CDPLShape.calcTanimotoComboScore,
        },
        "tversky_ref": {
            "shape": CDPLShape.calcReferenceShapeTverskyScore,
            "color": CDPLShape.calcReferenceColorTverskyScore,
            "combo": CDPLShape.calcReferenceTverskyComboScore,
        },
        "tversky_fit": {
            "shape": CDPLShape.calcAlignedShapeTverskyScore,
            "color": CDPLShape.calcAlignedColorTverskyScore,
            "combo": CDPLShape.calcAlignedTverskyComboScore,
        },
    }
else:
    _CDPKIT_SCORE_FUNCS = {}


@dataclass
class CDPKitWorkerContext:
    """Immutable context for CDPKit scoring workers.

    This dataclass encapsulates all state needed by worker processes,
    replacing module-level global variables. It is sent once per worker
    via the Pool initializer, not with each task.

    Attributes:
        reference_shapes: List of CDPKit GaussianShape objects for alignment.
        group_to_indices: Mapping from group index to reference shape indices.
        conf_file: Path to the conformer SDF file to score.
        opt_mode: Optimization/score mode ("shape" | "combo" | "color").
        score_variant: Normalization ("tanimoto" | "tversky_ref" | "tversky_fit").
        color_ff: Color pharmacophore typing ("default" | "oe_mills_dean").
    """

    reference_shapes: List
    group_to_indices: List[List[int]]
    conf_file: str
    opt_mode: str = "combo"
    score_variant: str = "tanimoto"
    color_ff: str = "default"


class CDPKitScoringWorker:
    """Callable worker for scoring molecules in parallel.

    This class encapsulates the worker logic and holds the context
    that would otherwise be stored in global variables. When used
    with multiprocessing.Pool, the initializer sets up the context
    once per worker process.

    Usage with Pool:
        worker = CDPKitScoringWorker()
        with Pool(n_workers, initializer=worker.initialize,
                  initargs=(context,)) as pool:
            results = pool.map(worker, mol_ids)
    """

    _context: ClassVar[Optional[CDPKitWorkerContext]] = None

    @staticmethod
    def initialize(context: CDPKitWorkerContext) -> None:
        """Initialize worker with shared context.

        Called once per worker process by Pool's initializer.
        Stores context at class level within the worker process.

        Args:
            context: The CDPKitWorkerContext containing reference shapes and config.
        """
        CDPKitScoringWorker._context = context

    def __call__(self, mol_id: int) -> Tuple[int, List[float]]:
        """Score a single molecule.

        This method is called by pool.map() for each molecule ID.
        Accesses the context set up by initialize().

        Args:
            mol_id: Index of the molecule to score.

        Returns:
            Tuple of (mol_id, list of scores per group).
        """
        ctx = CDPKitScoringWorker._context
        if ctx is None:
            return mol_id, []

        num_groups = len(ctx.group_to_indices) if ctx.group_to_indices else 0
        if not ctx.conf_file or not os.path.exists(ctx.conf_file) or num_groups == 0:
            return mol_id, [0.0] * num_groups

        group_scores = [0.0] * num_groups
        try:
            # Per-molecule wall-clock cap: a pathological molecule whose shape alignment (or
            # other unbounded step) spins would otherwise wedge this worker — and via the
            # serial fallback the whole epoch (cell 615). On timeout score it 0 (invalid) and
            # let the worker survive so the rest of the batch completes. No-op unless
            # DRUGEX_SCORE_MOL_TIMEOUT is set.
            with molecule_time_limit(mol_timeout()):
                # Re-read SDF and process only conformers for this mol_id
                reader = CDPLChem.FileSDFMoleculeReader(ctx.conf_file)
                target_prefix = f"mol_{mol_id}+"
                while True:
                    m = CDPLChem.BasicMolecule()
                    if not reader.read(m):
                        break
                    try:
                        name = CDPLChem.getName(m)
                    except Exception:
                        continue
                    if not name or not name.startswith(target_prefix):
                        continue
                    # Generate shapes for all conformers of this record and evaluate best
                    query_shapes = _generate_shape_helper(m, ctx.color_ff)
                    if not query_shapes:
                        continue
                    for group_idx, ref_indices in enumerate(ctx.group_to_indices):
                        best = group_scores[group_idx]
                        for ref_idx in ref_indices:
                            ref_shape = ctx.reference_shapes[ref_idx]
                            for query_shape in query_shapes:
                                score = _align_and_score_helper(
                                    query_shape, ref_shape, ctx.opt_mode, ctx.score_variant
                                )
                                if score > best:
                                    best = score
                        group_scores[group_idx] = best
        except ScoreTimeout:
            return mol_id, [0.0] * num_groups
        except Exception:
            return mol_id, [0.0] * num_groups

        return mol_id, group_scores


def _generate_shape_helper(cdpkit_mol, color_ff: str = "default"):
    """Generate Gaussian shape(s) for a molecule.

    Returns a list of shapes to ensure all conformers are considered.
    If no shapes can be generated, returns an empty list. When ``color_ff="oe_mills_dean"``
    the colored shape is typed with the OpenEye ImplicitMillsDean SMARTS (Lever 1) instead of
    CDPKit's default pharmacophore perception.
    """
    try:
        CDPLPharm.prepareForPharmacophoreGeneration(cdpkit_mol)
        shape_gen = CDPLShape.GaussianShapeGenerator()
        shape_gen.generatePharmacophoreShape(True)
        if color_ff == "oe_mills_dean":
            shape_gen.setPharmacophoreGenerator(_get_oe_pharm_generator())
        # Enable multi-conformer mode so every available conformer contributes a shape
        shape_gen.multiConformerMode(True)
        shape_set = shape_gen.generate(cdpkit_mol)
        if shape_set.getSize() == 0:
            return []
        return [shape_set.getElement(i) for i in range(shape_set.getSize())]
    except (RuntimeError, ValueError):
        return []


def _select_cdpkit_score(alignment_result, opt_mode, score_variant):
    """Pick the CDPKit score function for (opt_mode x score_variant), from the module-level
    ``_CDPKIT_SCORE_FUNCS`` table (built once at import, not per call).

    score_variant:
      "tanimoto"    -> symmetric Tanimoto (default; the OE-parity metric).
      "tversky_ref" -> reference-weighted Tversky ("is the REFERENCE pharmacophore covered
                       by the candidate?"); forgives the candidate being larger than the
                       small CCR2 references (PheSA, Wahl 2024, 10.1021/acs.jcim.4c00516).
      "tversky_fit" -> aligned(fit)-weighted Tversky (candidate-centric).
    All are CDPKit-native and exact. (RDKit reaches the same Tversky on its native basis by
    recovering the overlap from its reported Tanimoto + sov/sof; OpenEye reads native Tversky
    report columns.) Values are validated in CDPKitROCSScorer.__init__; the ``.get`` fallbacks
    here are a defensive backstop for direct callers (tests).
    """
    variant = _CDPKIT_SCORE_FUNCS.get(score_variant, _CDPKIT_SCORE_FUNCS["tanimoto"])
    return variant.get(opt_mode, variant["combo"])(alignment_result)


def _align_and_score_helper(query_shape, ref_shape, opt_mode="combo", score_variant="tanimoto"):
    """Align two shapes and return the score for the requested optimization mode + variant.

    opt_mode:
      "shape" -> ShapeTanimoto (shape-center start seeding).
      "combo" -> TanimotoCombo (shape-center start seeding; production default).
      "color" -> ColorTanimoto with COLOR-feature-center start seeding. CDPKit's BFGS
                 optimizer has no color gradient (``calcColorOverlapGradient`` does not
                 exist), so color is approached by seeding alignment starts at
                 pharmacophore-feature centers and selecting the best color overlap --
                 the closest open-source approximation to OpenEye ROCS ``-optchem``.
    score_variant: "tanimoto" (default) | "tversky_ref" | "tversky_fit"; see
      ``_select_cdpkit_score``. The pose is optimized identically (Tversky changes only the
      reported normalization), so the best score is selected over the same alignment results.
    """
    try:
        aligner = CDPLShape.GaussianShapeAlignment()
        start_generator = CDPLShape.PrincipalAxesAlignmentStartGenerator()
        if opt_mode == "color":
            start_generator.genColorCenterStarts(True)
            start_generator.genForAlignedShapeCenters(True)
        aligner.setStartGenerator(start_generator)
        aligner.setMaxNumOptimizationIterations(MAX_OPTIMIZATION_ITERATIONS)
        aligner.setOptimizationStopGradient(OPTIMIZATION_STOP_GRADIENT)
        aligner.addReferenceShape(ref_shape)
        if not aligner.align(query_shape) or aligner.getNumResults() == 0:
            return 0.0
        best_score = 0.0
        for i in range(aligner.getNumResults()):
            score = _select_cdpkit_score(aligner.getResult(i), opt_mode, score_variant)
            best_score = max(best_score, score)
        return best_score
    except (RuntimeError, ValueError):
        return 0.0


class CDPKitROCSScorer(Scorer):
    """ROCS-like scoring using CDPKit Gaussian shapes.

    Features:
    - Multi-reference support with automatic Gaussian shape generation.
    - Best-score selection across conformers and reference shapes.
    - Optional progress reporting and multiprocessing.
    - Dict-based reference grouping for multi-target optimization.
    - SMILES deduplication for efficient batch processing.
    - Memory-efficient worker design for scalability.

    Attributes:
        conformer_generator: 3D conformer generator used for query molecules.
        group_definitions: List of (name, paths) tuples defining reference groups.
        group_names: List of reference group names.
        shape_generator: CDPKit GaussianShapeGenerator instance.
        start_generator: CDPKit PrincipalAxesAlignmentStartGenerator instance.
        reference_mols: Loaded CDPKit molecules containing reference conformers.
        reference_shapes: Pre-computed Gaussian shapes for all references.
        group_to_indices: List mapping group indices to reference indices.
        show_progress: Whether to print progress and warnings.
        n_jobs: Requested worker count (-1 maps to available CPUs).
        _is_supermol: True when initialized with a single reference file.
    """

    def __init__(
        self,
        conformer_generator: ConformerGenerator,
        references: Union[str, List[str], Dict[str, List[str]]],
        show_progress: bool = True,
        n_jobs: int = -1,
        opt_mode: str = "combo",
        score_variant: str = "tanimoto",
        color_ff: str = "default",
    ):
        """Build a CDPKit ROCS scorer.

        Args:
            conformer_generator: Conformer generator used to produce query conformers.
            references: Path(s) to SDF references or a dict mapping group names
                to lists of reference paths.
            show_progress: Enables stdout progress indicators when True.
            n_jobs: Number of worker processes (-1 uses all available CPUs).
            color_ff: Color pharmacophore typing: "default" = CDPKit
                DefaultPharmacophoreGenerator (legacy), "oe_mills_dean" = OpenEye
                ImplicitMillsDean SMARTS port (Lever 1). OPT-IN: the default leaves color
                scoring unchanged. Recalibrate thresholds before making it the default
                (the OE-ground-truth Spearman GATE is a separate compute step).

        Raises:
            ImportError: If CDPKit bindings are not available.
            TypeError: If reference input types are unsupported.
            FileNotFoundError: If a reference path does not exist.
            ValueError: If reference shapes cannot be generated.
        """
        super().__init__()

        if not CDPL_AVAILABLE:
            raise ImportError("CDPKit is required. Install with `pip install cdpkit`.")

        if opt_mode not in _VALID_OPT_MODES:
            raise ValueError(
                f"opt_mode must be one of {sorted(_VALID_OPT_MODES)}, got {opt_mode!r}"
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
        self.opt_mode = opt_mode
        self.score_variant = score_variant
        self.color_ff = color_ff
        self.show_progress = show_progress
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.n_jobs = capped_n_jobs(self.n_jobs)  # DRUGEX_SCORE_NJOBS bound (cell-615 mem hang)

        self.group_definitions = self._prepare_reference_groups(references)
        self.group_names = [name for name, _ in self.group_definitions]

        self.shape_generator = CDPLShape.GaussianShapeGenerator()
        self.shape_generator.generatePharmacophoreShape(True)
        self.shape_generator.multiConformerMode(False)
        if self.color_ff == "oe_mills_dean":  # type reference color with the OE FF too
            self.shape_generator.setPharmacophoreGenerator(_get_oe_pharm_generator())
        self.start_generator = CDPLShape.PrincipalAxesAlignmentStartGenerator()

        self.reference_mols: List = []
        self.reference_shapes: List = []
        self.group_to_indices: List[List[int]] = []
        self._load_reference_groups()

        self._is_supermol = (
            len(self.group_names) == 1 and len(self.group_to_indices[0]) == 1
        )

        if self.show_progress:
            print(
                f"CDPKit ROCS ready with {len(self.reference_shapes)} reference "
                "shape(s)."
            )

    def _prepare_reference_groups(
        self, references: Union[str, List[str], Dict[str, List[str]]]
    ) -> List[Tuple[str, List[str]]]:
        groups: List[Tuple[str, List[str]]] = []
        if isinstance(references, dict):
            for name, paths in references.items():
                normalized = self._normalize_reference_list(paths)
                groups.append((str(name), normalized))
        else:
            normalized = self._normalize_reference_list(references)
            groups.append((_DEFAULT_CDPKIT_GROUP_NAME, normalized))

        if not groups:
            raise ValueError("At least one reference group must be provided")
        return groups

    def _normalize_reference_list(
        self, refs: Union[str, List[str]]
    ) -> List[str]:
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list) or not refs:
            raise ValueError("Reference group cannot be empty")
        for item in refs:
            if not isinstance(item, str):
                raise TypeError("CDPKit references must be file paths")
        return refs

    def _load_reference_groups(self) -> None:
        for name, paths in self.group_definitions:
            group_indices: List[int] = []
            for path in paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Reference file not found: {path}")
                mols = self._load_reference_molecules(path)
                if not mols:
                    raise ValueError(f"No valid molecules loaded from {path}")
                for mol in mols:
                    shape = self._generate_gaussian_shape(mol)
                    if shape is None:
                        continue
                    group_indices.append(len(self.reference_shapes))
                    self.reference_mols.append(mol)
                    self.reference_shapes.append(shape)
            if not group_indices:
                raise ValueError(
                    f"Reference group '{name}' produced no valid Gaussian shapes"
                )
            self.group_to_indices.append(group_indices)

    def _load_reference_molecules(self, filepath: str) -> List:
        molecules = []
        try:
            reader = CDPLChem.FileSDFMoleculeReader(filepath)
            while True:
                mol = CDPLChem.BasicMolecule()
                if not reader.read(mol):
                    break
                if mol.getNumAtoms() > 0:
                    molecules.append(mol)
        except Exception as exc:
            if self.show_progress:
                print(f"Warning: failed to read {filepath}: {exc}")
        return molecules

    def _generate_gaussian_shape(self, mol) -> Optional[object]:
        try:
            CDPLPharm.prepareForPharmacophoreGeneration(mol)
            shape_set = self.shape_generator.generate(mol)
            if shape_set.getSize() == 0:
                return None
            return shape_set.getElement(0)
        except (RuntimeError, ValueError) as exc:
            if self.show_progress:
                print(f"Warning: shape generation failed: {exc}")
            return None

    def getKey(self) -> List[str]:
        # Distinguish scorers by non-default opt_mode / score_variant so keys don't collide
        # (defaults combo / tanimoto -> '' -> legacy keys unchanged).
        sfx = ""
        if self.opt_mode != "combo":
            sfx += f"_{self.opt_mode}"
        if self.score_variant != "tanimoto":
            sfx += f"_{self.score_variant}"
        if self.color_ff == "oe_mills_dean":
            sfx += "_oe"
        if (
            len(self.group_names) == 1
            and self.group_names[0] == _DEFAULT_CDPKIT_GROUP_NAME
        ):
            if self._is_supermol:
                return [f"CDPKit_ROCS_Supermol_TanimotoCombo{sfx}"]
            refs = len(self.reference_shapes)
            return [f"CDPKit_ROCS_Aggregate_{refs}refs_TanimotoCombo{sfx}"]
        return [f"CDPKit_{name}{sfx}" for name in self.group_names]

    def create_progress_bar(self, total, desc):
        if not self.show_progress:
            return None
        try:
            from tqdm import tqdm

            return tqdm(total=total, desc=desc)
        except ImportError:
            return None

    def getScores(self, mols, frags=None) -> np.ndarray:
        """Score molecules using CDPKit Gaussian shape alignment.

        Args:
            mols: List of molecules (RDKit Mol objects or SMILES strings).
            frags: Unused, kept for interface compatibility.

        Returns:
            Array of shape (len(mols), num_groups) with TanimotoCombo scores.
        """
        num_groups = len(self.group_to_indices)
        if num_groups == 0:
            raise ValueError("No reference groups configured")
        if not mols:
            return np.zeros((0, num_groups))

        smiles_list = []
        for mol in mols:
            if mol is None:
                smiles_list.append(None)
            elif isinstance(mol, str):
                smiles_list.append(mol)
            else:
                smiles_list.append(Chem.MolToSmiles(mol))

        unique_smiles, unique_to_original = self._deduplicate_smiles(smiles_list)
        if not unique_smiles:
            return np.zeros((len(mols), num_groups))

        # ignore_cleanup_errors: on networked scratch (NFS/Lustre) a lingering reader handle at
        # exit becomes a silly-rename (.nfsXXXX) so rmtree raises ENOTEMPTY *after* scores are
        # computed -> the result would be silently lost. Swallow the cleanup error (the dir is
        # reaped by the job's TMPDIR cleanup).
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            conf_file = self.conformer_generator.genConformers(unique_smiles, tmpdir)
            if not os.path.exists(conf_file):
                if self.show_progress:
                    print("Warning: conformer file not generated")
                return np.zeros((len(mols), num_groups))

            # Build quick presence map without holding molecules in memory
            present_ids = set()
            try:
                reader = CDPLChem.FileSDFMoleculeReader(conf_file)
                while True:
                    m = CDPLChem.BasicMolecule()
                    if not reader.read(m):
                        break
                    try:
                        name = CDPLChem.getName(m)
                        mol_id = int(name.split("+")[0].split("_")[1])
                        present_ids.add(mol_id)
                    except Exception:
                        continue
            except Exception as exc:
                if self.show_progress:
                    print(f"Warning: failed to read conformers: {exc}")
                return np.zeros((len(mols), num_groups))

            num_unique = len(unique_smiles)
            scores_unique = np.zeros((num_unique, num_groups), dtype=np.float32)

            # Create worker context and callable worker instance
            context = CDPKitWorkerContext(
                reference_shapes=self.reference_shapes,
                group_to_indices=self.group_to_indices,
                conf_file=conf_file,
                opt_mode=self.opt_mode,
                score_variant=self.score_variant,
                color_ff=self.color_ff,
            )
            worker = CDPKitScoringWorker()

            if self.n_jobs == 1:
                # Sequential mode: initialize and call worker directly
                worker.initialize(context)
                for mol_id in range(num_unique):
                    if mol_id not in present_ids:
                        continue
                    _, group_scores = worker(mol_id)
                    if len(group_scores) == num_groups:
                        scores_unique[mol_id] = np.asarray(group_scores, dtype=np.float32)
            else:
                # Parallel mode: use Pool with worker initializer
                worker_args = [mol_id for mol_id in range(num_unique) if mol_id in present_ids]
                effective_jobs = max(1, self.n_jobs)
                chunksize = max(1, len(worker_args) // (effective_jobs * 4))
                timeout = score_timeout()  # batch-level backstop; None = wait forever
                try:
                    with pool_context().Pool(
                        self.n_jobs,
                        initializer=CDPKitScoringWorker.initialize,
                        initargs=(context,),
                    ) as pool:
                        if self.show_progress:
                            try:
                                from tqdm import tqdm

                                results = list(
                                    tqdm(
                                        pool.imap(worker, worker_args, chunksize=chunksize),
                                        total=len(worker_args),
                                        desc=f"Scoring with {self.getKey()}",
                                    )
                                )
                            except ImportError:
                                results = pool.map_async(
                                    worker, worker_args, chunksize=chunksize
                                ).get(timeout=timeout)
                        else:
                            # map_async(...).get(timeout) bounds the batch: if a molecule hangs
                            # in a pure-C step the per-molecule SIGALRM can't interrupt, the get
                            # times out and the with-exit terminates the wedged worker.
                            results = pool.map_async(
                                worker, worker_args, chunksize=chunksize
                            ).get(timeout=timeout)
                    for mol_id, group_scores in results:
                        if 0 <= mol_id < num_unique and len(group_scores) == num_groups:
                            scores_unique[mol_id] = np.asarray(group_scores, dtype=np.float32)
                except MPTimeoutError:
                    # Batch backstop fired: a worker was wedged in non-interruptible C and the
                    # with-exit terminated it. Do NOT serial-fallback (it would re-hit the same
                    # hang in the main process). Leave the unscored molecules at 0 (-> 0 reward).
                    if self.show_progress:
                        print(
                            f"Warning: CDPKit scoring timed out after {timeout}s; "
                            f"{len(worker_args)} molecules left unscored (->0)"
                        )
                except Exception as exc:
                    if self.show_progress:
                        print(
                            f"Warning: parallel processing failed ({exc}), "
                            "switching to sequential mode"
                        )
                    # Transient parallel failure (NOT a hang). The serial path is bounded
                    # per-molecule by the worker's SIGALRM (molecule_time_limit), so it cannot
                    # hang either.
                    worker.initialize(context)
                    for mol_id in range(num_unique):
                        if mol_id not in present_ids:
                            continue
                        _, group_scores = worker(mol_id)
                        if len(group_scores) == num_groups:
                            scores_unique[mol_id] = np.asarray(group_scores, dtype=np.float32)

            scores = np.zeros((len(mols), num_groups), dtype=np.float32)
            for unique_id, original_indices in unique_to_original.items():
                for original_idx in original_indices:
                    scores[original_idx] = scores_unique[unique_id]

            return scores

    @staticmethod
    def _deduplicate_smiles(
        smiles_list: List[Optional[str]],
    ) -> Tuple[List[str], Dict[int, List[int]]]:
        """Return unique SMILES plus mapping back to originals."""
        unique_smiles: List[str] = []
        unique_lookup: Dict[str, int] = {}
        unique_to_original: Dict[int, List[int]] = defaultdict(list)

        for idx, smi in enumerate(smiles_list):
            if smi is None:
                continue
            existing = unique_lookup.get(smi)
            if existing is None:
                existing = len(unique_smiles)
                unique_smiles.append(smi)
                unique_lookup[smi] = existing
            unique_to_original[existing].append(idx)

        return unique_smiles, unique_to_original
