"""
RDKit-based ROCS-like scoring using shape and color Tanimoto similarity.

This module provides open-source alternatives to OpenEye ROCS using RDKit's
rdShapeAlign module for shape and pharmacophore-based molecular similarity.

Classes:
    RDKitROCSScorer: Unified scorer accepting single or multiple references
    RDKitAggregateScorer: Legacy wrapper (deprecated)
    RDKitSupermoleculeScorer: Legacy wrapper (deprecated)

Migration Guide:
    # Old code
    scorer = RDKitAggregateScorer(conf_gen, [mol1, mol2, mol3])

    # New code (recommended)
    scorer = RDKitROCSScorer(conf_gen, [mol1, mol2, mol3])

    # Old code
    scorer = RDKitSupermoleculeScorer(conf_gen, "supermol.sdf")

    # New code (recommended)
    scorer = RDKitROCSScorer(conf_gen, "supermol.sdf")
"""

import os
import tempfile
from collections import defaultdict
from typing import List, Union
from multiprocessing import Pool, cpu_count

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdShapeAlign

from drugex.training.scorers.interfaces import ConformerGenerator, Scorer


# Module-level worker function for multiprocessing (must be picklable)
def _score_molecule_rdkit_worker(args):
    """Worker function for parallel RDKit ROCS scoring.

    Handles both single and multiple reference molecules by accepting
    a list of references (even if it contains only one element).

    Args:
        args: Tuple of (mol_id, mol_conformers, reference_mols,
                       score_type, use_colors)

    Returns:
        Tuple of (mol_id, max_score)
    """
    mol_id, mol_conformers, reference_mols, score_type, use_colors = args

    if not mol_conformers:
        return mol_id, 0.0

    max_score = 0.0
    try:
        for conf_mol in mol_conformers:
            for ref_mol in reference_mols:
                # Calculate shape score
                if conf_mol is None or ref_mol is None:
                    continue

                if conf_mol.GetNumConformers() == 0 or ref_mol.GetNumConformers() == 0:
                    continue

                # Try all conformer combinations
                for query_conf in conf_mol.GetConformers():
                    for ref_conf in ref_mol.GetConformers():
                        try:
                            # Make a copy of the query molecule
                            probe_copy = Chem.Mol(conf_mol)

                            # Perform alignment
                            result = rdShapeAlign.AlignMol(
                                ref_mol,
                                probe_copy,
                                refConfId=ref_conf.GetId(),
                                probeConfId=query_conf.GetId(),
                                useColors=use_colors,
                            )

                            # Extract scores
                            if isinstance(result, (list, tuple)) and len(result) >= 2:
                                shape_score = result[0]
                                color_score = result[1]
                            else:
                                continue

                            # Calculate final score based on score_type
                            if score_type == "shape":
                                score = shape_score
                            elif score_type == "color":
                                score = color_score
                            else:  # TanimotoCombo
                                score = shape_score + color_score

                            max_score = max(max_score, score)

                        except (RuntimeError, ValueError):
                            continue
    except Exception:
        # Return 0.0 for any unexpected errors
        return mol_id, 0.0

    return mol_id, max_score


class RDKitROCSScorer(Scorer):
    """Unified RDKit ROCS scorer supporting single or multiple references.

    This scorer unifies the functionality of RDKitAggregateScorer and
    RDKitSupermoleculeScorer. It accepts references in multiple formats:
    - Single file path (str): Loads one reference molecule
    - Multiple file paths (List[str]): Loads multiple references
    - Single molecule (Chem.Mol): Uses one reference
    - Multiple molecules (List[Chem.Mol]): Uses multiple references

    Attributes:
        conformer_generator: ConformerGenerator for generating 3D conformers
        reference_mols: List of reference molecules with conformers
        score_type: Type of score - "TanimotoCombo", "shape", or "color"
        use_colors: Whether to use pharmacophore colors in alignment
        show_progress: Whether to display progress messages
        n_jobs: Number of parallel jobs
    """

    def __init__(
        self,
        conformer_generator: ConformerGenerator,
        references: Union[str, List[str], Chem.Mol, List[Chem.Mol]],
        score_type: str = "TanimotoCombo",
        use_colors: bool = True,
        show_progress: bool = True,
        n_jobs: int = -1,
    ):
        """Initialize the unified RDKit ROCS scorer.

        Args:
            conformer_generator: Instance for generating 3D conformers
            references: Reference molecule(s) in any of these formats:
                - str: Path to SDF file with single reference
                - List[str]: Paths to SDF files with multiple references
                - Chem.Mol: Single RDKit molecule with conformers
                - List[Chem.Mol]: Multiple RDKit molecules with conformers
            score_type: Type of score - "TanimotoCombo", "shape", or "color"
            use_colors: Whether to use pharmacophore colors
            show_progress: Whether to show progress during scoring
            n_jobs: Number of parallel jobs (-1 for all CPUs)

        Raises:
            TypeError: If references type is not supported
            ValueError: If references is empty or invalid
            FileNotFoundError: If reference file path does not exist
        """
        super().__init__()

        self.conformer_generator = conformer_generator
        self.score_type = score_type
        self.use_colors = use_colors
        self.show_progress = show_progress
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

        # Detect if this is supermolecule mode (before normalization)
        self._is_supermol = self._detect_supermol_mode(references)

        # Normalize references to List[Chem.Mol]
        self.reference_mols = self._normalize_references(references)

        # Validate
        self._validate_references()

    def _detect_supermol_mode(
        self,
        references: Union[str, List[str], Chem.Mol, List[Chem.Mol]]
    ) -> bool:
        """Detect if this scorer was initialized in supermolecule mode.

        Args:
            references: Original references argument

        Returns:
            True if single reference, False if multiple
        """
        if isinstance(references, (str, Chem.Mol)):
            return True
        if isinstance(references, list) and len(references) == 1:
            return True
        return False

    def _normalize_references(
        self,
        references: Union[str, List[str], Chem.Mol, List[Chem.Mol]]
    ) -> List[Chem.Mol]:
        """Normalize various reference formats to List[Chem.Mol].

        Args:
            references: References in any supported format

        Returns:
            List of RDKit molecules with conformers

        Raises:
            TypeError: If references type is not supported
            FileNotFoundError: If file path does not exist
            ValueError: If molecules cannot be loaded
        """
        # Case 1: Single file path
        if isinstance(references, str):
            return [self._load_molecule_from_file(references)]

        # Case 2: Multiple file paths
        if isinstance(references, list) and all(isinstance(r, str) for r in references):
            return [self._load_molecule_from_file(path) for path in references]

        # Case 3: Single molecule
        if isinstance(references, Chem.Mol):
            return [references]

        # Case 4: Multiple molecules
        if isinstance(references, list) and all(isinstance(r, Chem.Mol) for r in references):
            return references

        # Case 5: Invalid
        raise TypeError(
            f"references must be str, List[str], Chem.Mol, or List[Chem.Mol], "
            f"got {type(references)}"
        )

    def _load_molecule_from_file(self, path: str) -> Chem.Mol:
        """Load molecule from SDF file.

        Args:
            path: Path to SDF file

        Returns:
            RDKit molecule with conformers

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file cannot be read or is empty
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Reference file not found: {path}")

        try:
            suppl = Chem.SDMolSupplier(path, removeHs=False)
            if not suppl:
                raise ValueError(f"Could not open SDF file: {path}")

            mol = next(iter(suppl), None)
            if mol is None:
                raise ValueError(f"No molecules found in file: {path}")

            return mol
        except Exception as e:
            if self.show_progress:
                print(f"Error loading molecule from {path}: {e}")
            raise ValueError(f"Failed to load molecule from {path}: {e}")

    def _validate_references(self):
        """Validate that all reference molecules have conformers.

        Raises:
            ValueError: If any reference is invalid
        """
        if not self.reference_mols:
            raise ValueError("At least one reference molecule is required")

        for i, ref_mol in enumerate(self.reference_mols):
            if ref_mol is None:
                raise ValueError(f"Reference molecule at index {i} is None")
            if ref_mol.GetNumConformers() == 0:
                raise ValueError(
                    f"Reference molecule at index {i} has no conformers. "
                    "Generate conformers before creating the scorer."
                )

    def getKey(self) -> str:
        """Return scorer identifier.

        Returns:
            str: Identifier distinguishing supermol vs aggregate mode
        """
        if self._is_supermol:
            return f"RDKit_Supermol_{self.score_type}"
        else:
            n_refs = len(self.reference_mols)
            return f"RDKit_Aggregate_{n_refs}refs_{self.score_type}"

    def _calculate_shape_score(self, query_mol: Chem.Mol, ref_mol: Chem.Mol) -> float:
        """Calculate the best shape score between query and reference.

        Tries all conformer combinations and returns the maximum score.

        Args:
            query_mol: Query molecule with conformers
            ref_mol: Reference molecule with conformers

        Returns:
            Maximum score across all conformer combinations
        """
        if query_mol is None or ref_mol is None:
            return 0.0

        if query_mol.GetNumConformers() == 0 or ref_mol.GetNumConformers() == 0:
            return 0.0

        best_score = 0.0

        # Try all conformer combinations
        for query_conf in query_mol.GetConformers():
            for ref_conf in ref_mol.GetConformers():
                try:
                    # Make a copy of the query molecule (probe is modified during alignment)
                    probe_copy = Chem.Mol(query_mol)

                    # Perform alignment and get shape and color scores
                    # Returns tuple: (shape_tanimoto, color_tanimoto)
                    result = rdShapeAlign.AlignMol(
                        ref_mol,
                        probe_copy,
                        refConfId=ref_conf.GetId(),
                        probeConfId=query_conf.GetId(),
                        useColors=self.use_colors,
                    )

                    # Extract scores from result
                    if isinstance(result, (list, tuple)) and len(result) >= 2:
                        shape_score = result[0]
                        color_score = result[1]
                    else:
                        # Unexpected result format, skip
                        continue

                    # Calculate final score based on score_type
                    if self.score_type == "shape":
                        score = shape_score
                    elif self.score_type == "color":
                        score = color_score
                    else:  # TanimotoCombo (default)
                        # Tanimoto Combo is the sum of shape and color Tanimoto
                        score = shape_score + color_score

                    best_score = max(best_score, score)

                except (RuntimeError, ValueError) as e:
                    # Alignment can fail for various reasons - skip this combination
                    if self.show_progress:
                        print(f"Warning: Shape alignment failed: {e}")
                    continue

        return best_score

    def getScores(self, mols: List[Chem.Mol], frags=None) -> np.ndarray:
        """Score molecules against reference molecules.

        Args:
            mols: List of RDKit molecules or SMILES strings to score
            frags: Optional fragments (not used in this scorer)

        Returns:
            Array of scores with shape (len(mols), 1)
        """
        if not mols:
            return np.zeros((0, 1))

        num_mols = len(mols)
        scores = np.zeros((num_mols, 1))

        if self.show_progress:
            print(f"Scoring {num_mols} molecules with {self.getKey()}...")

        # Convert molecules to SMILES for uniform conformer generation
        smiles_list = []
        for mol in mols:
            if mol is None:
                smiles_list.append(None)
            elif isinstance(mol, str):
                smiles_list.append(mol)
            else:
                try:
                    smiles_list.append(Chem.MolToSmiles(mol))
                except:
                    smiles_list.append(None)

        # Generate conformers using the conformer generator
        with tempfile.TemporaryDirectory() as tmpdir:
            conf_file = self.conformer_generator.genConformers(smiles_list, tmpdir)

            # Load generated conformers if file exists
            if not os.path.exists(conf_file):
                if self.show_progress:
                    print(f"Warning: Conformer generation failed, returning zero scores")
                return scores

            # Read conformers and group by molecule ID
            conformers_by_mol = defaultdict(list)
            try:
                suppl = Chem.SDMolSupplier(conf_file, removeHs=False)
                for conf_mol in suppl:
                    if conf_mol is None:
                        continue
                    # Extract molecule ID from name (format: "mol_<id>+<isomer>")
                    try:
                        name = conf_mol.GetProp("_Name")
                        mol_id = int(name.split("+")[0].split("_")[1])
                        conformers_by_mol[mol_id].append(conf_mol)
                    except:
                        continue
            except:
                if self.show_progress:
                    print(f"Warning: Failed to load conformers from {conf_file}")
                return scores

            # Score each molecule using its conformers
            if self.n_jobs == 1:
                # Sequential processing
                for mol_id in range(num_mols):
                    mol_conformers = conformers_by_mol.get(mol_id, [])

                    if not mol_conformers:
                        scores[mol_id] = 0.0
                        continue

                    # Calculate maximum score across all conformers and reference molecules
                    max_score = 0.0
                    for conf_mol in mol_conformers:
                        for ref_mol in self.reference_mols:
                            score = self._calculate_shape_score(conf_mol, ref_mol)
                            max_score = max(max_score, score)

                    scores[mol_id] = max_score

                    # Progress indication
                    if self.show_progress and (mol_id + 1) % 100 == 0:
                        print(f"  Scored {mol_id + 1}/{num_mols} molecules")
            else:
                # Parallel processing
                # Prepare arguments for workers
                worker_args = [
                    (mol_id, conformers_by_mol.get(mol_id, []), self.reference_mols,
                     self.score_type, self.use_colors)
                    for mol_id in range(num_mols)
                ]

                # Calculate optimal chunk size
                chunksize = max(1, num_mols // (self.n_jobs * 4))

                try:
                    with Pool(self.n_jobs) as pool:
                        if self.show_progress:
                            # Use imap for progress tracking
                            try:
                                from tqdm import tqdm
                                results = list(tqdm(
                                    pool.imap(_score_molecule_rdkit_worker, worker_args, chunksize=chunksize),
                                    total=num_mols,
                                    desc="Scoring molecules"
                                ))
                            except ImportError:
                                # Fallback without progress bar
                                results = pool.map(_score_molecule_rdkit_worker, worker_args, chunksize=chunksize)
                                print(f"  Scored {num_mols} molecules (parallel)")
                        else:
                            results = pool.map(_score_molecule_rdkit_worker, worker_args, chunksize=chunksize)

                    # Collect results
                    for mol_id, max_score in results:
                        scores[mol_id] = max_score

                except Exception as e:
                    if self.show_progress:
                        print(f"Warning: Parallel processing failed ({e}), falling back to sequential")
                    # Fallback to sequential processing
                    for mol_id in range(num_mols):
                        mol_conformers = conformers_by_mol.get(mol_id, [])
                        if not mol_conformers:
                            scores[mol_id] = 0.0
                            continue
                        max_score = 0.0
                        for conf_mol in mol_conformers:
                            for ref_mol in self.reference_mols:
                                score = self._calculate_shape_score(conf_mol, ref_mol)
                                max_score = max(max_score, score)
                        scores[mol_id] = max_score

        if self.show_progress:
            print(f"Scoring complete. Average score: {scores.mean():.3f}")

        return scores


class RDKitAggregateScorer(RDKitROCSScorer):
    """Legacy aggregate scorer - use RDKitROCSScorer instead.

    This class exists for backward compatibility. New code should use
    RDKitROCSScorer directly.

    .. deprecated::
        Use :class:`RDKitROCSScorer` instead. This wrapper will be removed
        in a future version.
    """

    def __init__(
        self,
        conformer_generator: ConformerGenerator,
        reference_mols: List[Chem.Mol],
        score_type: str = "TanimotoCombo",
        use_colors: bool = True,
        show_progress: bool = True,
        n_jobs: int = -1,
    ):
        """Initialize aggregate scorer (legacy interface).

        Args:
            conformer_generator: ConformerGenerator instance
            reference_mols: List of reference molecules with conformers
            score_type: Type of score - "TanimotoCombo", "shape", or "color"
            use_colors: Whether to use pharmacophore colors
            show_progress: Whether to show progress
            n_jobs: Number of parallel jobs
        """
        import warnings
        warnings.warn(
            "RDKitAggregateScorer is deprecated. Use RDKitROCSScorer instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # Delegate to unified implementation
        super().__init__(
            conformer_generator=conformer_generator,
            references=reference_mols,
            score_type=score_type,
            use_colors=use_colors,
            show_progress=show_progress,
            n_jobs=n_jobs,
        )

    def getKey(self) -> str:
        """Return scorer identifier (legacy format)."""
        return f"RDKit_Aggregate_{self.score_type}"


class RDKitSupermoleculeScorer(RDKitROCSScorer):
    """Legacy supermolecule scorer - use RDKitROCSScorer instead.

    This class exists for backward compatibility. New code should use
    RDKitROCSScorer directly.

    .. deprecated::
        Use :class:`RDKitROCSScorer` instead. This wrapper will be removed
        in a future version.
    """

    def __init__(
        self,
        conformer_generator: ConformerGenerator,
        supermol_file: str,
        score_type: str = "TanimotoCombo",
        use_colors: bool = True,
        show_progress: bool = True,
        n_jobs: int = -1,
    ):
        """Initialize supermolecule scorer (legacy interface).

        Args:
            conformer_generator: ConformerGenerator instance
            supermol_file: Path to SDF file containing supermolecule
            score_type: Type of score - "TanimotoCombo", "shape", or "color"
            use_colors: Whether to use pharmacophore colors
            show_progress: Whether to show progress
            n_jobs: Number of parallel jobs
        """
        import warnings
        warnings.warn(
            "RDKitSupermoleculeScorer is deprecated. Use RDKitROCSScorer instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # Delegate to unified implementation
        super().__init__(
            conformer_generator=conformer_generator,
            references=supermol_file,  # Single file path
            score_type=score_type,
            use_colors=use_colors,
            show_progress=show_progress,
            n_jobs=n_jobs,
        )

    def getKey(self) -> str:
        """Return scorer identifier (legacy format)."""
        return f"RDKit_Supermolecule_{self.score_type}"
