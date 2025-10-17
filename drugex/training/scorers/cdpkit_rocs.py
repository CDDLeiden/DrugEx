"""CDPKit-based ROCS scorer implementation."""

import os
import tempfile
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Union

import numpy as np

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


def _generate_shape_helper(cdpkit_mol):
    """Generate a GaussianShape for use in multiprocessing."""
    try:
        CDPLPharm.prepareForPharmacophoreGeneration(cdpkit_mol)
        shape_gen = CDPLShape.GaussianShapeGenerator()
        shape_gen.generatePharmacophoreShape(True)
        shape_gen.multiConformerMode(False)
        shape_set = shape_gen.generate(cdpkit_mol)
        if shape_set.getSize() == 0:
            return None
        return shape_set.getElement(0)
    except (RuntimeError, ValueError):
        return None


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


def _score_molecule_cdpkit_worker(args):
    """Score a single molecule in a worker process."""
    mol_id, cdpkit_mols, reference_shapes = args
    if not cdpkit_mols:
        return mol_id, 0.0

    max_score = 0.0
    try:
        for cdpkit_mol in cdpkit_mols:
            query_shape = _generate_shape_helper(cdpkit_mol)
            if query_shape is None:
                continue
            for ref_shape in reference_shapes:
                score = _align_and_score_helper(query_shape, ref_shape)
                max_score = max(max_score, score)
    except Exception:
        return mol_id, 0.0

    return mol_id, max_score


class CDPKitROCSScorer(Scorer):
    """ROCS-like scoring using CDPKit Gaussian shapes.

    Features:
    - Multi-reference support with automatic Gaussian shape generation.
    - Best-score selection across conformers and reference shapes.
    - Optional progress reporting and multiprocessing.

    Attributes:
        conformer_generator: 3D conformer generator used for query molecules.
        reference_mol_files: Normalized list of reference SDF file paths.
        reference_mols: Loaded CDPKit molecules containing reference conformers.
        reference_shapes: Pre-computed Gaussian shapes for all references.
        show_progress: Whether to print progress and warnings.
        n_jobs: Requested worker count (-1 maps to available CPUs).
        _is_supermol: True when initialized with a single reference file.
    """

    def __init__(
        self,
        conformer_generator: ConformerGenerator,
        references: Union[str, List[str]],
        show_progress: bool = True,
        n_jobs: int = -1,
    ):
        """Build a CDPKit ROCS scorer.

        Args:
            conformer_generator: Conformer generator used to produce query conformers.
            references: Path or paths to SDF references with 3D coordinates.
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
        self._is_supermol = isinstance(references, str)
        self.reference_mol_files = self._normalize_references(references)

        self.shape_generator = CDPLShape.GaussianShapeGenerator()
        self.shape_generator.generatePharmacophoreShape(True)
        self.shape_generator.multiConformerMode(False)
        self.start_generator = CDPLShape.PrincipalAxesAlignmentStartGenerator()

        self.reference_mols: List = []
        self.reference_shapes: List = []
        self._load_references()

        if self.show_progress:
            print(
                f"CDPKit ROCS ready with {len(self.reference_shapes)} reference "
                "shape(s)."
            )

    def _normalize_references(self, references: Union[str, List[str]]) -> List[str]:
        if isinstance(references, str):
            return [references]
        if isinstance(references, list) and all(isinstance(r, str) for r in references):
            return references
        raise TypeError("references must be str or List[str]")

    def _load_references(self):
        for ref_file in self.reference_mol_files:
            if not os.path.exists(ref_file):
                raise FileNotFoundError(f"Reference file not found: {ref_file}")

            mols = self._load_reference_molecules(ref_file)
            if not mols:
                raise ValueError(f"No valid molecules loaded from {ref_file}")

            for mol in mols:
                shape = self._generate_gaussian_shape(mol)
                if shape is not None:
                    self.reference_mols.append(mol)
                    self.reference_shapes.append(shape)

        if not self.reference_shapes:
            raise ValueError("No reference shapes generated from provided files")

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
        if self._is_supermol:
            return ["CDPKit_ROCS_Supermol_TanimotoCombo"]
        refs = len(self.reference_shapes)
        return [f"CDPKit_ROCS_Aggregate_{refs}refs_TanimotoCombo"]

    def create_progress_bar(self, total, desc):
        if not self.show_progress:
            return None
        try:
            from tqdm import tqdm

            return tqdm(total=total, desc=desc)
        except ImportError:
            return None

    def getScores(self, mols, frags=None) -> np.ndarray:
        if not mols:
            return np.zeros((0, 1))

        from rdkit import Chem

        smiles_list = []
        for mol in mols:
            if mol is None:
                smiles_list.append(None)
            elif isinstance(mol, str):
                smiles_list.append(mol)
            else:
                smiles_list.append(Chem.MolToSmiles(mol))

        with tempfile.TemporaryDirectory() as tmpdir:
            conf_file = self.conformer_generator.genConformers(smiles_list, tmpdir)
            if not os.path.exists(conf_file):
                if self.show_progress:
                    print("Warning: conformer file not generated")
                return np.zeros((len(mols), 1))

            conformers_by_mol = {}
            try:
                reader = CDPLChem.FileSDFMoleculeReader(conf_file)
                while True:
                    cdpkit_mol = CDPLChem.BasicMolecule()
                    if not reader.read(cdpkit_mol):
                        break
                    try:
                        name = CDPLChem.getName(cdpkit_mol)
                        mol_id = int(name.split("+")[0].split("_")[1])
                        conformers_by_mol.setdefault(mol_id, []).append(cdpkit_mol)
                    except Exception:
                        continue
            except Exception as exc:
                if self.show_progress:
                    print(f"Warning: failed to read conformers: {exc}")
                return np.zeros((len(mols), 1))

            num_mols = len(mols)
            scores = np.zeros((num_mols, 1), dtype=np.float32)
            conformers = [conformers_by_mol.get(i, []) for i in range(num_mols)]

            if self.n_jobs == 1:
                for mol_id, mol_conformers in enumerate(conformers):
                    max_score = 0.0
                    for cdpkit_mol in mol_conformers:
                        query_shape = self._generate_gaussian_shape(cdpkit_mol)
                        if query_shape is None:
                            continue
                        for ref_shape in self.reference_shapes:
                            score = self._align_and_score(query_shape, ref_shape)
                            max_score = max(max_score, score)
                    scores[mol_id] = max_score
            else:
                worker_args = [
                    (mol_id, mol_conformers, self.reference_shapes)
                    for mol_id, mol_conformers in enumerate(conformers)
                ]
                effective_jobs = max(1, self.n_jobs)
                chunksize = max(1, len(conformers) // (effective_jobs * 4))
                try:
                    with Pool(self.n_jobs) as pool:
                        if self.show_progress:
                            try:
                                from tqdm import tqdm

                                results = list(
                                    tqdm(
                                        pool.imap(
                                            _score_molecule_cdpkit_worker,
                                            worker_args,
                                            chunksize=chunksize,
                                        ),
                                        total=len(conformers),
                                        desc=f"Scoring with {self.getKey()}",
                                    )
                                )
                            except ImportError:
                                results = pool.map(
                                    _score_molecule_cdpkit_worker,
                                    worker_args,
                                    chunksize=chunksize,
                                )
                                print(
                                    f"  Scored {len(conformers)} molecules "
                                    "(parallel)"
                                )
                        else:
                            results = pool.map(
                                _score_molecule_cdpkit_worker,
                                worker_args,
                                chunksize=chunksize,
                            )
                    for mol_id, max_score in results:
                        scores[mol_id] = max_score
                except Exception as exc:
                    if self.show_progress:
                        print(
                            f"Warning: parallel processing failed ({exc}), "
                            "switching to sequential mode"
                        )
                    for mol_id, mol_conformers in enumerate(conformers):
                        max_score = 0.0
                        for cdpkit_mol in mol_conformers:
                            query_shape = self._generate_gaussian_shape(cdpkit_mol)
                            if query_shape is None:
                                continue
                            for ref_shape in self.reference_shapes:
                                score = self._align_and_score(
                                    query_shape, ref_shape
                                )
                                max_score = max(max_score, score)
                        scores[mol_id] = max_score

            return scores
