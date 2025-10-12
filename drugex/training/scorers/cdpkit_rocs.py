"""
CDPKit ROCS-like Scoring Implementation.

This module provides shape-based molecular similarity scoring using CDPKit's
GaussianShape alignment, implementing a ROCS-like TanimotoCombo scoring approach.

The TanimotoCombo score combines:
- Shape Tanimoto: measures 3D shape overlap
- Color Tanimoto: measures pharmacophoric feature overlap
- Final score is in range 0-2 (raw TanimotoCombo score)

Implementation based on CDPKit's official shape alignment example:
https://github.com/molinfo-vienna/CDPKit/blob/master/Examples/Python/align_mols_by_shape.py

Default settings match CDPKit shapescreen CLI tool defaults:
https://cdpkit.org/applications/shapescreen.html#synopsis

Key settings aligned with shapescreen defaults:
- Screening Mode: BEST_PER_QUERY
- Scoring Function: TANIMOTO_COMBO
- Shape Center Starts: Enabled (-S flag, default=true)
- Atom Center Starts: Disabled (-A flag, default=false)
- Color Center Starts: Disabled (-C flag, default=false)
- Random Starts: 0 (-R flag, default=0)
- All Carbon Mode: Not used (-W flag - investigated but not currently implemented)
- Overlay Optimization: Enabled (-a flag, default=true, via GaussianShapeAlignment.align)
- Max Iterations: 500 (optimization parameter)
- Stop Gradient: 0.001 (optimization parameter)

References:
- CDPKit Shape API: https://cdpkit.org/cdpl_api_doc/python_api_doc/namespaceCDPL_1_1Shape.html
- CDPKit Python Tutorial: https://cdpkit.org/cdpl_python_tutorial/index.html
- CDPKit Python Cookbook: https://cdpkit.org/cdpl_python_cookbook/index.html

Classes:
    CDPKitROCSScorer: Unified scorer accepting single or multiple references
    CDPKitROCSAggregateScorer: Legacy wrapper (deprecated)
    CDPKitROCSSupermoleculeScorer: Legacy wrapper (deprecated)

Requirements:
    - CDPKit must be installed: pip install cdpkit
    - Reference molecules must have 3D coordinates
    - ConformerGenerator must produce molecules with valid 3D conformers

Migration Guide:
    # Old code
    scorer = CDPKitROCSAggregateScorer(conf_gen, ["ref1.sdf", "ref2.sdf"])

    # New code (recommended)
    scorer = CDPKitROCSScorer(conf_gen, ["ref1.sdf", "ref2.sdf"])

    # Old code
    scorer = CDPKitROCSSupermoleculeScorer(conf_gen, "supermol.sdf")

    # New code (recommended)
    scorer = CDPKitROCSScorer(conf_gen, "supermol.sdf")
"""

import os
import tempfile
from typing import List, Optional, Union
from multiprocessing import Pool, cpu_count

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

# Constants for shape alignment optimization
# These values match CDPKit C++ ScreeningSettings defaults
# Source: CDPKit source code Shape::ScreeningSettings::ScreeningSettings()
# Official defaults: numOptIter=20, optStopGrad=1.0, greedyOpt=true
MAX_OPTIMIZATION_ITERATIONS = 20  # Maximum iterations for gradient-based optimization (CDPKit default)
OPTIMIZATION_STOP_GRADIENT = 1.0  # Gradient threshold for convergence (CDPKit default)

# Gaussian shape generation parameters
# These are CDPKit defaults for GaussianShapeGenerator
SHAPE_RADIUS = -1.0  # Automatic radius calculation (default)
INCLUDE_HYDROGENS = False  # Exclude hydrogens from shape (default)
SHAPE_HARDNESS = 2.7  # Shape hardness parameter (default)


# Module-level helper functions for multiprocessing
def _generate_shape_helper(cdpkit_mol):
    """Generate GaussianShape for a CDPKit molecule (picklable helper)."""
    try:
        # Prepare molecule for pharmacophore generation
        CDPLPharm.prepareForPharmacophoreGeneration(cdpkit_mol)

        # Create shape generator
        shape_gen = CDPLShape.GaussianShapeGenerator()
        shape_gen.generatePharmacophoreShape(True)
        shape_gen.multiConformerMode(False)

        # Generate shape
        shape_set = shape_gen.generate(cdpkit_mol)
        if shape_set.getSize() == 0:
            return None

        return shape_set.getElement(0)
    except (RuntimeError, ValueError):
        return None


def _align_and_score_helper(query_shape, ref_shape):
    """Align and score two GaussianShapes (picklable helper)."""
    try:
        aligner = CDPLShape.GaussianShapeAlignment()

        # Set up start generator
        start_generator = CDPLShape.PrincipalAxesAlignmentStartGenerator()
        aligner.setStartGenerator(start_generator)

        # Set optimization parameters
        aligner.setMaxNumOptimizationIterations(MAX_OPTIMIZATION_ITERATIONS)
        aligner.setOptimizationStopGradient(OPTIMIZATION_STOP_GRADIENT)

        # Add reference and align
        aligner.addReferenceShape(ref_shape)

        if not aligner.align(query_shape):
            return 0.0

        if aligner.getNumResults() == 0:
            return 0.0

        # Get best score across all results
        best_score = 0.0
        for i in range(aligner.getNumResults()):
            alignment_result = aligner.getResult(i)
            combo_score = CDPLShape.calcTanimotoComboScore(alignment_result)
            best_score = max(best_score, combo_score)

        return best_score
    except (RuntimeError, ValueError):
        return 0.0


def _score_molecule_cdpkit_worker(args):
    """Worker function for parallel CDPKit ROCS scoring.

    Handles both single and multiple reference shapes by accepting
    a list of shapes (even if it contains only one element).

    Args:
        args: Tuple of (mol_id, cdpkit_mols, reference_shapes)

    Returns:
        Tuple of (mol_id, max_score)
    """
    mol_id, cdpkit_mols, reference_shapes = args

    if not cdpkit_mols:
        return mol_id, 0.0

    max_score = 0.0
    try:
        for cdpkit_mol in cdpkit_mols:
            # Generate shape for this conformer
            query_shape = _generate_shape_helper(cdpkit_mol)
            if query_shape is None:
                continue

            # Score against all reference shapes
            for ref_shape in reference_shapes:
                score = _align_and_score_helper(query_shape, ref_shape)
                max_score = max(max_score, score)
    except Exception:
        return mol_id, 0.0

    return mol_id, max_score


class CDPKitROCSScorer(Scorer):
    """Unified CDPKit ROCS scorer supporting single or multiple references.

    This scorer unifies CDPKitROCSAggregateScorer and
    CDPKitROCSSupermoleculeScorer functionality. It accepts:
    - Single file path (str): Loads one reference
    - Multiple file paths (List[str]): Loads multiple references

    Attributes:
        conformer_generator: ConformerGenerator for 3D conformers
        reference_mol_files: List of paths to reference SDF files
        reference_mols: Loaded CDPKit reference molecules
        reference_shapes: Pre-computed GaussianShapes
        show_progress: Whether to display progress
        n_jobs: Number of parallel jobs
    """

    def __init__(
        self,
        conformer_generator: ConformerGenerator,
        references: Union[str, List[str]],
        show_progress: bool = True,
        n_jobs: int = -1,
    ):
        """Initialize the CDPKit ROCS scorer.

        Args:
            conformer_generator: ConformerGenerator instance for creating 3D conformers
            references: Reference molecule(s) in any of these formats:
                - str: Path to SDF file with single reference
                - List[str]: Paths to SDF files with multiple references
            show_progress: If True, display progress information during scoring
            n_jobs: Number of parallel jobs (-1 for all CPUs)

        Raises:
            ImportError: If CDPKit is not available
            TypeError: If references type is not supported
            FileNotFoundError: If any reference file does not exist
            ValueError: If reference molecules cannot be loaded
        """
        super().__init__()

        if not CDPL_AVAILABLE:
            raise ImportError(
                "CDPKit is required but not available. "
                "Install with: pip install cdpkit"
            )

        self.conformer_generator = conformer_generator
        self.show_progress = show_progress
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

        # Detect if this is supermolecule mode (before normalization)
        self._is_supermol = isinstance(references, str)

        # Normalize to List[str]
        self.reference_mol_files = self._normalize_references(references)

        # Initialize shape generator
        self.shape_generator = CDPLShape.GaussianShapeGenerator()
        self.shape_generator.generatePharmacophoreShape(True)  # Enable color features
        self.shape_generator.multiConformerMode(False)  # One shape per molecule

        # Initialize start generator (CRITICAL FOR PROPER ALIGNMENT!)
        #
        # These settings match shapescreen CLI defaults to ensure comparable scores.
        # Source: https://cdpkit.org/applications/shapescreen.html#synopsis
        #
        # The PrincipalAxesAlignmentStartGenerator creates initial alignment poses by:
        # 1. Computing principal axes for both query and reference molecules
        # 2. Aligning molecules along their principal axes
        # 3. Placing shape centers at the coordinate system origin
        #
        # Without proper starting poses, optimization gets stuck in local minima,
        # resulting in scores 20-40% lower than optimal.
        #
        # CDPKit 1.2.3 default settings already match CLI defaults:
        # - shapeCenterStarts: True (matches -S --shape-center-starts)
        # - colorCenterStarts: False (matches -C --color-center-starts)
        # - nonColorCenterStarts: False (matches -A --atom-center-starts)
        # - randomStarts: False (matches -R --random-starts=0)
        self.start_generator = CDPLShape.PrincipalAxesAlignmentStartGenerator()

        # Verify default settings match CLI (these should already be correct)
        if self.show_progress:
            print(f"Start generator settings: shape={self.start_generator.shapeCenterStarts}, "
                  f"color={self.start_generator.colorCenterStarts}, "
                  f"nonColor={self.start_generator.nonColorCenterStarts}, "
                  f"random={self.start_generator.randomStarts}")

        # Load and validate references
        self.reference_mols = []
        self.reference_shapes = []
        self._load_references()

        if self.show_progress:
            print(f"CDPKit ROCS Scorer initialized with {len(self.reference_shapes)} reference shapes")

    def _normalize_references(self, references: Union[str, List[str]]) -> List[str]:
        """Normalize reference input to List[str] of file paths.

        Args:
            references: Single path or list of paths

        Returns:
            List of file paths

        Raises:
            TypeError: If references is not str or List[str]
        """
        if isinstance(references, str):
            return [references]
        if isinstance(references, list) and all(isinstance(r, str) for r in references):
            return references
        raise TypeError(
            f"references must be str or List[str], got {type(references)}"
        )

    def _load_references(self):
        """Load all reference molecules and generate shapes.

        Raises:
            FileNotFoundError: If any reference file does not exist
            ValueError: If molecules cannot be loaded or shapes cannot be generated
        """
        for ref_file in self.reference_mol_files:
            if not os.path.exists(ref_file):
                raise FileNotFoundError(f"Reference file not found: {ref_file}")

            # Load molecules from file
            mols = self._load_reference_molecules(ref_file)
            if not mols:
                raise ValueError(f"No valid molecules loaded from: {ref_file}")

            # Generate shapes for all reference molecules
            for mol in mols:
                shape = self._generate_gaussian_shape(mol)
                if shape is not None:
                    self.reference_mols.append(mol)
                    self.reference_shapes.append(shape)

        if not self.reference_shapes:
            raise ValueError(
                "No valid reference shapes could be generated from provided files"
            )

    def _load_reference_molecules(self, filepath: str) -> List:
        """Load reference molecules from an SDF file.

        Args:
            filepath: Path to SDF file containing reference molecules

        Returns:
            List of CDPKit molecules with 3D coordinates
        """
        molecules = []
        try:
            reader = CDPLChem.FileSDFMoleculeReader(filepath)
            while True:
                mol = CDPLChem.BasicMolecule()
                if not reader.read(mol):
                    break
                if mol.getNumAtoms() > 0:
                    molecules.append(mol)
        except Exception as e:
            if self.show_progress:
                print(f"Warning: Error reading {filepath}: {e}")
        return molecules

    def _generate_gaussian_shape(self, mol) -> Optional:
        """Generate a GaussianShape for a CDPKit molecule with pharmacophore features.

        This method prepares the molecule for pharmacophore generation and creates
        a Gaussian shape that includes both shape and color (pharmacophore) features.
        This is essential for accurate TanimotoCombo scoring.

        Based on CDPKit example:
        https://github.com/molinfo-vienna/CDPKit/blob/master/Examples/Python/align_mols_by_shape.py

        Args:
            mol: CDPKit molecule with 3D coordinates

        Returns:
            GaussianShape object with pharmacophore features or None if generation fails
        """
        try:
            # Prepare molecule for pharmacophore generation (color features)
            # This is critical for TanimotoCombo scoring!
            CDPLPharm.prepareForPharmacophoreGeneration(mol)

            # Generate shape with pharmacophore features using the shape generator
            # The generator returns a GaussianShapeSet
            shape_set = self.shape_generator.generate(mol)

            # Extract the first (and only) shape from the shape set
            if shape_set.getSize() == 0:
                return None

            return shape_set.getElement(0)
        except (RuntimeError, ValueError) as e:
            if self.show_progress:
                print(f"Shape generation failed: {e}")
            return None

    def _align_and_score(self, query_shape, ref_shape) -> float:
        """Align query shape to reference shape and calculate TanimotoCombo score.

        The TanimotoCombo score combines shape Tanimoto and color Tanimoto,
        providing a comprehensive similarity measure that accounts for both
        3D shape overlap and pharmacophoric feature alignment.

        Alignment methodology based on CDPKit example:
        https://github.com/molinfo-vienna/CDPKit/blob/master/Examples/Python/align_mols_by_shape.py

        Settings match CDPKit shapescreen CLI defaults:
        https://cdpkit.org/applications/shapescreen.html#synopsis

        Args:
            query_shape: GaussianShape of query molecule
            ref_shape: GaussianShape of reference molecule

        Returns:
            Raw TanimotoCombo score in range [0, 2]
        """
        try:
            aligner = CDPLShape.GaussianShapeAlignment()

            # CRITICAL: Set the starting pose generator (matches shapescreen CLI defaults)
            # Source: https://cdpkit.org/applications/shapescreen.html#synopsis
            #
            # The shapescreen CLI uses PrincipalAxesAlignmentStartGenerator with
            # shape-center-starts enabled by default (-S flag). This generates
            # multiple intelligent starting poses based on principal axes alignment,
            # dramatically improving the chance of finding the global optimum.
            #
            # Without this, alignment may get stuck in local minima resulting in
            # scores 20-40% lower than expected.
            aligner.setStartGenerator(self.start_generator)

            # Set optimization parameters (match shapescreen defaults)
            aligner.setMaxNumOptimizationIterations(MAX_OPTIMIZATION_ITERATIONS)
            aligner.setOptimizationStopGradient(OPTIMIZATION_STOP_GRADIENT)

            # Add reference shape to aligner
            aligner.addReferenceShape(ref_shape)

            # Perform alignment with starting poses (corresponds to -a --opt-overlay=true)
            if not aligner.align(query_shape):
                return 0.0

            if aligner.getNumResults() == 0:
                return 0.0

            # Get best alignment result across all starting poses
            # The aligner may generate multiple results from different starting poses,
            # we select the one with the best TanimotoCombo score
            best_score = 0.0
            for i in range(aligner.getNumResults()):
                alignment_result = aligner.getResult(i)

                # Calculate TanimotoCombo score (shape + color)
                # This is equivalent to shapescreen -s TANIMOTO_COMBO
                combo_score = CDPLShape.calcTanimotoComboScore(alignment_result)

                # Return raw TanimotoCombo score (range 0-2)
                best_score = max(best_score, combo_score)

            return best_score

        except (RuntimeError, ValueError) as e:
            if self.show_progress:
                print(f"Alignment failed: {e}")
            return 0.0

    def getKey(self) -> str:
        """Return scorer identifier for result reporting.

        Returns:
            str: Scorer key distinguishing supermol vs aggregate mode
        """
        if self._is_supermol:
            return "CDPKit_ROCS_Supermol_TanimotoCombo"
        else:
            n_refs = len(self.reference_shapes)
            return f"CDPKit_ROCS_Aggregate_{n_refs}refs_TanimotoCombo"

    def create_progress_bar(self, total, desc):
        """Create a progress bar if show_progress is enabled."""
        if self.show_progress:
            try:
                from tqdm import tqdm
                return tqdm(total=total, desc=desc)
            except ImportError:
                return None
        return None

    def getScores(self, mols, frags=None) -> np.ndarray:
        """Score molecules against reference shapes using CDPKit ROCS.

        Args:
            mols: List of RDKit molecules or SMILES strings
            frags: Optional fragments (not used)

        Returns:
            np.ndarray: Scores with shape (num_molecules, 1)
        """
        if not mols:
            return np.zeros((0, 1))

        # Convert to SMILES if needed
        from rdkit import Chem
        smiles_list = []
        for mol in mols:
            if mol is None:
                smiles_list.append(None)
            elif isinstance(mol, str):
                smiles_list.append(mol)
            else:  # RDKit molecule
                smiles_list.append(Chem.MolToSmiles(mol))

        # Generate conformers using the conformer generator
        with tempfile.TemporaryDirectory() as tmpdir:
            conf_file = self.conformer_generator.genConformers(smiles_list, tmpdir)

            # Load conformers directly with CDPKit
            if not os.path.exists(conf_file):
                if self.show_progress:
                    print("Warning: Conformer file not generated")
                return np.zeros((len(mols), 1))

            # Group CDPKit conformers by molecule ID
            conformers_by_mol = {}
            try:
                reader = CDPLChem.FileSDFMoleculeReader(conf_file)
                while True:
                    cdpkit_mol = CDPLChem.BasicMolecule()
                    if not reader.read(cdpkit_mol):
                        break

                    # Extract molecule ID from name
                    try:
                        name = CDPLChem.getName(cdpkit_mol)
                        mol_id = int(name.split("+")[0].split("_")[1])
                    except (IndexError, ValueError, RuntimeError):
                        continue

                    if mol_id not in conformers_by_mol:
                        conformers_by_mol[mol_id] = []
                    conformers_by_mol[mol_id].append(cdpkit_mol)

            except Exception as e:
                if self.show_progress:
                    print(f"Error reading conformers: {e}")
                return np.zeros((len(mols), 1))

            # Create ordered list of conformers for scoring
            conformers = [conformers_by_mol.get(i, []) for i in range(len(mols))]

            # Score conformers directly (no RDKit conversion needed)
            scores = self._score_cdpkit_conformers(conformers)

        return scores

    def _score_cdpkit_conformers(
        self, conformers: List[List]
    ) -> np.ndarray:
        """Score molecules by aligning CDPKit conformers against reference shapes.

        For each molecule:
        1. Generate GaussianShapes for each CDPKit conformer
        2. Align against all reference shapes
        3. Return maximum TanimotoCombo score

        Args:
            conformers: List of CDPKit conformer lists for each molecule

        Returns:
            np.ndarray: Scores with shape (num_molecules, 1)
        """
        scores = np.zeros((len(conformers), 1))

        if self.n_jobs == 1:
            # Sequential processing
            progress_bar = self.create_progress_bar(
                total=len(conformers), desc=f"Scoring with {self.getKey()}"
            )

            for mol_id, mol_conformers in enumerate(conformers):
                max_score = 0.0

                if not mol_conformers:
                    scores[mol_id] = max_score
                    if progress_bar:
                        progress_bar.update(1)
                    continue

                # Score each CDPKit conformer against all references
                for cdpkit_mol in mol_conformers:
                    # Generate shape for this conformer (already CDPKit)
                    query_shape = self._generate_gaussian_shape(cdpkit_mol)
                    if query_shape is None:
                        continue

                    # Score against all reference shapes
                    for ref_shape in self.reference_shapes:
                        score = self._align_and_score(query_shape, ref_shape)
                        if score > max_score:
                            max_score = score

                scores[mol_id] = max_score
                if progress_bar:
                    progress_bar.update(1)

            if progress_bar:
                progress_bar.close()
        else:
            # Parallel processing
            # Prepare arguments for workers
            worker_args = [
                (mol_id, conformers[mol_id], self.reference_shapes)
                for mol_id in range(len(conformers))
            ]

            # Calculate optimal chunk size
            chunksize = max(1, len(conformers) // (self.n_jobs * 4))

            try:
                with Pool(self.n_jobs) as pool:
                    if self.show_progress:
                        # Use imap for progress tracking
                        try:
                            from tqdm import tqdm
                            results = list(tqdm(
                                pool.imap(_score_molecule_cdpkit_worker, worker_args, chunksize=chunksize),
                                total=len(conformers),
                                desc=f"Scoring with {self.getKey()}"
                            ))
                        except ImportError:
                            # Fallback without progress bar
                            results = pool.map(_score_molecule_cdpkit_worker, worker_args, chunksize=chunksize)
                            print(f"  Scored {len(conformers)} molecules (parallel)")
                    else:
                        results = pool.map(_score_molecule_cdpkit_worker, worker_args, chunksize=chunksize)

                # Collect results
                for mol_id, max_score in results:
                    scores[mol_id] = max_score

            except Exception as e:
                if self.show_progress:
                    print(f"Warning: Parallel processing failed ({e}), falling back to sequential")
                # Fallback to sequential processing
                for mol_id, mol_conformers in enumerate(conformers):
                    max_score = 0.0
                    if not mol_conformers:
                        scores[mol_id] = max_score
                        continue
                    for cdpkit_mol in mol_conformers:
                        query_shape = self._generate_gaussian_shape(cdpkit_mol)
                        if query_shape is None:
                            continue
                        for ref_shape in self.reference_shapes:
                            score = self._align_and_score(query_shape, ref_shape)
                            max_score = max(max_score, score)
                    scores[mol_id] = max_score

        return scores


class CDPKitROCSAggregateScorer(CDPKitROCSScorer):
    """Legacy aggregate scorer - use CDPKitROCSScorer instead.

    This class exists for backward compatibility. New code should use
    CDPKitROCSScorer directly.

    .. deprecated::
        Use :class:`CDPKitROCSScorer` instead. This wrapper will be removed
        in a future version.
    """

    def __init__(
        self,
        conformer_generator: ConformerGenerator,
        reference_mols: List[str],
        show_progress: bool = True,
        n_jobs: int = -1,
    ):
        """Initialize aggregate scorer (legacy interface).

        Args:
            conformer_generator: ConformerGenerator instance
            reference_mols: List of paths to reference SDF files
            show_progress: Whether to display progress
            n_jobs: Number of parallel jobs
        """
        import warnings
        warnings.warn(
            "CDPKitROCSAggregateScorer is deprecated. Use CDPKitROCSScorer instead.",
            DeprecationWarning,
            stacklevel=2
        )

        super().__init__(
            conformer_generator=conformer_generator,
            references=reference_mols,
            show_progress=show_progress,
            n_jobs=n_jobs,
        )

    def getKey(self) -> str:
        """Return scorer identifier (legacy format)."""
        return "CDPKit_ROCS_TanimotoCombo"


class CDPKitROCSSupermoleculeScorer(CDPKitROCSScorer):
    """Legacy supermolecule scorer - use CDPKitROCSScorer instead.

    This class exists for backward compatibility. New code should use
    CDPKitROCSScorer directly.

    .. deprecated::
        Use :class:`CDPKitROCSScorer` instead. This wrapper will be removed
        in a future version.
    """

    def __init__(
        self,
        conformer_generator: ConformerGenerator,
        supermol_file: str,
        show_progress: bool = True,
        n_jobs: int = -1,
    ):
        """Initialize supermolecule scorer (legacy interface).

        Args:
            conformer_generator: ConformerGenerator instance
            supermol_file: Path to supermolecule SDF file
            show_progress: Whether to display progress
            n_jobs: Number of parallel jobs
        """
        import warnings
        warnings.warn(
            "CDPKitROCSSupermoleculeScorer is deprecated. Use CDPKitROCSScorer instead.",
            DeprecationWarning,
            stacklevel=2
        )

        super().__init__(
            conformer_generator=conformer_generator,
            references=supermol_file,  # Single file
            show_progress=show_progress,
            n_jobs=n_jobs,
        )

    def getKey(self) -> str:
        """Return scorer identifier (legacy format)."""
        return "CDPKit_ROCS_Supermol_TanimotoCombo"
