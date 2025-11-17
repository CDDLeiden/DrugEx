"""CDPKit-based ROCS scorer implementation."""

import os
import tempfile
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple, Union

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

MAX_OPTIMIZATION_ITERATIONS = 20
OPTIMIZATION_STOP_GRADIENT = 1.0

_CDPKIT_WORKER_SETTINGS: Dict[str, object] = {}
_DEFAULT_CDPKIT_GROUP_NAME = "_default_group"


def _cdpkit_worker_init(reference_shapes, group_to_indices, conf_file: str):
    """Initializer to avoid sending reference shapes with each task."""
    global _CDPKIT_WORKER_SETTINGS
    _CDPKIT_WORKER_SETTINGS = {
        "reference_shapes": reference_shapes,
        "group_to_indices": group_to_indices,
        "conf_file": conf_file,
    }


def _generate_shape_helper(cdpkit_mol):
    """Generate Gaussian shape(s) for a molecule.

    Returns a list of shapes to ensure all conformers are considered.
    If no shapes can be generated, returns an empty list.
    """
    try:
        CDPLPharm.prepareForPharmacophoreGeneration(cdpkit_mol)
        shape_gen = CDPLShape.GaussianShapeGenerator()
        shape_gen.generatePharmacophoreShape(True)
        # Enable multi-conformer mode so every available conformer contributes a shape
        shape_gen.multiConformerMode(True)
        shape_set = shape_gen.generate(cdpkit_mol)
        if shape_set.getSize() == 0:
            return []
        return [shape_set.getElement(i) for i in range(shape_set.getSize())]
    except (RuntimeError, ValueError):
        return []


def _align_and_score_helper(query_shape, ref_shape):
    """Align two shapes and return the best TanimotoCombo score."""
    try:
        aligner = CDPLShape.GaussianShapeAlignment()
        start_generator = CDPLShape.PrincipalAxesAlignmentStartGenerator()
        aligner.setStartGenerator(start_generator)
        aligner.setMaxNumOptimizationIterations(MAX_OPTIMIZATION_ITERATIONS)
        aligner.setOptimizationStopGradient(OPTIMIZATION_STOP_GRADIENT)
        aligner.addReferenceShape(ref_shape)
        if not aligner.align(query_shape) or aligner.getNumResults() == 0:
            return 0.0
        best_score = 0.0
        for i in range(aligner.getNumResults()):
            alignment_result = aligner.getResult(i)
            score = CDPLShape.calcTanimotoComboScore(alignment_result)
            best_score = max(best_score, score)
        return best_score
    except (RuntimeError, ValueError):
        return 0.0


def _score_molecule_cdpkit_worker(mol_id: int):
    """Score a single molecule in a worker process by reloading conformers on demand."""
    reference_shapes = _CDPKIT_WORKER_SETTINGS.get("reference_shapes", [])
    group_to_indices = _CDPKIT_WORKER_SETTINGS.get("group_to_indices", [])
    conf_file = _CDPKIT_WORKER_SETTINGS.get("conf_file", None)
    num_groups = len(group_to_indices) if group_to_indices else 0
    if not conf_file or not os.path.exists(conf_file) or num_groups == 0:
        return mol_id, [0.0] * num_groups

    group_scores = [0.0] * num_groups
    try:
        # Re-read SDF and process only conformers for this mol_id
        reader = CDPLChem.FileSDFMoleculeReader(conf_file)
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
            query_shapes = _generate_shape_helper(m)
            if not query_shapes:
                continue
            for group_idx, ref_indices in enumerate(group_to_indices):
                best = group_scores[group_idx]
                for ref_idx in ref_indices:
                    ref_shape = reference_shapes[ref_idx]
                    for query_shape in query_shapes:
                        score = _align_and_score_helper(query_shape, ref_shape)
                        if score > best:
                            best = score
                group_scores[group_idx] = best
    except Exception:
        return mol_id, [0.0] * num_groups

    return mol_id, group_scores


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
    ):
        """Build a CDPKit ROCS scorer.

        Args:
            conformer_generator: Conformer generator used to produce query conformers.
            references: Path(s) to SDF references or a dict mapping group names
                to lists of reference paths.
            show_progress: Enables stdout progress indicators when True.
            n_jobs: Number of worker processes (-1 uses all available CPUs).

        Raises:
            ImportError: If CDPKit bindings are not available.
            TypeError: If reference input types are unsupported.
            FileNotFoundError: If a reference path does not exist.
            ValueError: If reference shapes cannot be generated.
        """
        super().__init__()

        if not CDPL_AVAILABLE:
            raise ImportError("CDPKit is required. Install with `pip install cdpkit`.")

        self.conformer_generator = conformer_generator
        self.show_progress = show_progress
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

        self.group_definitions = self._prepare_reference_groups(references)
        self.group_names = [name for name, _ in self.group_definitions]

        self.shape_generator = CDPLShape.GaussianShapeGenerator()
        self.shape_generator.generatePharmacophoreShape(True)
        self.shape_generator.multiConformerMode(False)
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

    def _align_and_score(self, query_shape, ref_shape) -> float:
        try:
            aligner = CDPLShape.GaussianShapeAlignment()
            aligner.setStartGenerator(self.start_generator)
            aligner.setMaxNumOptimizationIterations(MAX_OPTIMIZATION_ITERATIONS)
            aligner.setOptimizationStopGradient(OPTIMIZATION_STOP_GRADIENT)
            aligner.addReferenceShape(ref_shape)
            if not aligner.align(query_shape) or aligner.getNumResults() == 0:
                return 0.0
            best_score = 0.0
            for i in range(aligner.getNumResults()):
                result = aligner.getResult(i)
                score = CDPLShape.calcTanimotoComboScore(result)
                best_score = max(best_score, score)
            return best_score
        except (RuntimeError, ValueError) as exc:
            if self.show_progress:
                print(f"Warning: alignment failed: {exc}")
            return 0.0

    def getKey(self) -> List[str]:
        if (
            len(self.group_names) == 1
            and self.group_names[0] == _DEFAULT_CDPKIT_GROUP_NAME
        ):
            if self._is_supermol:
                return ["CDPKit_ROCS_Supermol_TanimotoCombo"]
            refs = len(self.reference_shapes)
            return [f"CDPKit_ROCS_Aggregate_{refs}refs_TanimotoCombo"]
        return [f"CDPKit_{name}" for name in self.group_names]

    def create_progress_bar(self, total, desc):
        if not self.show_progress:
            return None
        try:
            from tqdm import tqdm

            return tqdm(total=total, desc=desc)
        except ImportError:
            return None

    def getScores(self, mols, frags=None) -> np.ndarray:
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

        with tempfile.TemporaryDirectory() as tmpdir:
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

            if self.n_jobs == 1:
                # Initialize worker globals for sequential scoring as well
                _cdpkit_worker_init(
                    self.reference_shapes, self.group_to_indices, conf_file
                )
                for mol_id in range(num_unique):
                    if mol_id not in present_ids:
                        continue
                    _, group_scores = _score_molecule_cdpkit_worker(mol_id)
                    if len(group_scores) == num_groups:
                        scores_unique[mol_id] = np.asarray(group_scores, dtype=np.float32)
                # Clear worker settings
                _CDPKIT_WORKER_SETTINGS = {}
            else:
                worker_args = [mol_id for mol_id in range(num_unique) if mol_id in present_ids]
                effective_jobs = max(1, self.n_jobs)
                chunksize = max(1, len(worker_args) // (effective_jobs * 4))
                try:
                    with Pool(
                        self.n_jobs,
                        initializer=_cdpkit_worker_init,
                        initargs=(self.reference_shapes, self.group_to_indices, conf_file),
                    ) as pool:
                        if self.show_progress:
                            try:
                                from tqdm import tqdm

                                results = list(
                                    tqdm(
                                        pool.imap(_score_molecule_cdpkit_worker, worker_args, chunksize=chunksize),
                                        total=len(worker_args),
                                        desc=f"Scoring with {self.getKey()}",
                                    )
                                )
                            except ImportError:
                                results = pool.map(_score_molecule_cdpkit_worker, worker_args, chunksize=chunksize)
                                print(
                                    f"  Scored {len(worker_args)} molecules "
                                    "(parallel)"
                                )
                        else:
                            results = pool.map(_score_molecule_cdpkit_worker, worker_args, chunksize=chunksize)
                    for mol_id, group_scores in results:
                        if 0 <= mol_id < num_unique and len(group_scores) == num_groups:
                            scores_unique[mol_id] = np.asarray(group_scores, dtype=np.float32)
                except Exception as exc:
                    if self.show_progress:
                        print(
                            f"Warning: parallel processing failed ({exc}), "
                            "switching to sequential mode"
                        )
                    for mol_id in range(num_unique):
                        if mol_id not in present_ids:
                            continue
                        _, group_scores = _score_molecule_cdpkit_worker(mol_id)
                        if len(group_scores) == num_groups:
                            scores_unique[mol_id] = np.asarray(group_scores, dtype=np.float32)
                finally:
                    _CDPKIT_WORKER_SETTINGS = {}

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
