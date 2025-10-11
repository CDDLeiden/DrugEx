"""
RDKit-based ROCS-like scoring using shape and color Tanimoto similarity.

This module provides open-source alternatives to OpenEye ROCS using RDKit's
rdShapeAlign module for shape and pharmacophore-based molecular similarity.
"""

from typing import List, Optional
from multiprocessing import Pool, cpu_count

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdShapeAlign

from drugex.training.scorers.interfaces import ConformerGenerator, Scorer


# Module-level worker function for multiprocessing (must be picklable)
def _score_molecule_aggregate_worker(args):
    """
    Worker function for parallel molecule scoring in RDKitAggregateScorer.

    Args:
        args: Tuple of (mol_id, mol_conformers, reference_mols, score_type, use_colors)

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


def _score_molecule_supermol_worker(args):
    """
    Worker function for parallel molecule scoring in RDKitSupermoleculeScorer.

    Args:
        args: Tuple of (mol_id, mol_conformers, reference_mol, score_type, use_colors)

    Returns:
        Tuple of (mol_id, max_score)
    """
    mol_id, mol_conformers, reference_mol, score_type, use_colors = args

    if not mol_conformers:
        return mol_id, 0.0

    max_score = 0.0
    try:
        for conf_mol in mol_conformers:
            # Calculate shape score against supermolecule
            if conf_mol is None or reference_mol is None:
                continue

            if conf_mol.GetNumConformers() == 0 or reference_mol.GetNumConformers() == 0:
                continue

            # Try all conformer combinations
            for query_conf in conf_mol.GetConformers():
                for ref_conf in reference_mol.GetConformers():
                    try:
                        # Make a copy of the query molecule
                        probe_copy = Chem.Mol(conf_mol)

                        # Perform alignment
                        result = rdShapeAlign.AlignMol(
                            reference_mol,
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


class RDKitAggregateScorer(Scorer):
    """
    Scores molecules based on shape and color Tanimoto similarity to multiple reference molecules.

    Returns the maximum Tanimoto Combo score (shape + color) across all reference molecules
    and all conformer combinations.

    Attributes:
        conformer_generator: ConformerGenerator instance for generating conformers
        reference_mols: List of reference molecules with pre-generated conformers
        score_type: Type of score to return - "TanimotoCombo" (default), "shape", or "color"
        use_colors: Whether to use pharmacophore colors in alignment (default: True)
        show_progress: Whether to display progress messages (default: True)
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
        """
        Initialize the RDKit aggregate scorer.

        Args:
            conformer_generator: ConformerGenerator instance for generating conformers
            reference_mols: List of reference molecules with conformers
            score_type: Type of score - "TanimotoCombo", "shape", or "color"
            use_colors: Whether to use pharmacophore colors in alignment
            show_progress: Whether to show progress during scoring
            n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential, >1 for specific count)
        """
        super().__init__()

        self.conformer_generator = conformer_generator
        self.reference_mols = reference_mols
        self.score_type = score_type
        self.use_colors = use_colors
        self.show_progress = show_progress
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

        # Validate reference molecules
        if not reference_mols:
            raise ValueError("At least one reference molecule is required")

        # Ensure all reference molecules have conformers
        for i, ref_mol in enumerate(reference_mols):
            if ref_mol is None:
                raise ValueError(f"Reference molecule at index {i} is None")
            if ref_mol.GetNumConformers() == 0:
                raise ValueError(
                    f"Reference molecule at index {i} has no conformers. "
                    "Generate conformers before creating the scorer."
                )

    def getKey(self) -> str:
        """Return scorer identifier."""
        return f"RDKit_Aggregate_{self.score_type}"

    def _calculate_shape_score(self, query_mol: Chem.Mol, ref_mol: Chem.Mol) -> float:
        """
        Calculate the best shape score between a query molecule and a reference molecule.

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
        """
        Score molecules against all reference molecules.

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
        import os
        import tempfile
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
            from collections import defaultdict
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
                                    pool.imap(_score_molecule_aggregate_worker, worker_args, chunksize=chunksize),
                                    total=num_mols,
                                    desc="Scoring molecules"
                                ))
                            except ImportError:
                                # Fallback without progress bar
                                results = pool.map(_score_molecule_aggregate_worker, worker_args, chunksize=chunksize)
                                print(f"  Scored {num_mols} molecules (parallel)")
                        else:
                            results = pool.map(_score_molecule_aggregate_worker, worker_args, chunksize=chunksize)

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


class RDKitSupermoleculeScorer(Scorer):
    """
    Scores molecules based on shape and color Tanimoto similarity to a single supermolecule reference.

    The supermolecule is typically a merged representation of multiple active compounds,
    representing the common pharmacophoric features.

    Attributes:
        conformer_generator: ConformerGenerator instance for generating conformers
        supermol_file: Path to SDF file containing the supermolecule
        score_type: Type of score to return - "TanimotoCombo" (default), "shape", or "color"
        use_colors: Whether to use pharmacophore colors in alignment (default: True)
        show_progress: Whether to display progress messages (default: True)
        reference_mol: Loaded supermolecule with conformers
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
        """
        Initialize the RDKit supermolecule scorer.

        Args:
            conformer_generator: ConformerGenerator instance for generating conformers
            supermol_file: Path to SDF file containing the supermolecule
            score_type: Type of score - "TanimotoCombo", "shape", or "color"
            use_colors: Whether to use pharmacophore colors in alignment
            show_progress: Whether to show progress during scoring
            n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential, >1 for specific count)

        Raises:
            ValueError: If supermolecule file cannot be loaded or has no conformers
        """
        super().__init__()

        self.conformer_generator = conformer_generator
        self.supermol_file = supermol_file
        self.score_type = score_type
        self.use_colors = use_colors
        self.show_progress = show_progress
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

        # Load and validate supermolecule
        self.reference_mol = self._load_supermolecule(supermol_file)

        if self.reference_mol is None:
            raise ValueError(f"Could not load supermolecule from {supermol_file}")

        if self.reference_mol.GetNumConformers() == 0:
            raise ValueError(
                f"Supermolecule from {supermol_file} has no conformers. "
                "Generate conformers before creating the scorer."
            )

    def getKey(self) -> str:
        """Return scorer identifier."""
        return f"RDKit_Supermolecule_{self.score_type}"

    def _load_supermolecule(self, path: str) -> Optional[Chem.Mol]:
        """
        Load supermolecule from SDF file.

        Args:
            path: Path to SDF file

        Returns:
            First molecule from the file, or None if loading fails
        """
        try:
            suppl = Chem.SDMolSupplier(path, removeHs=False)
            if not suppl:
                return None

            # Get first molecule from supplier
            mol = next(iter(suppl), None)
            return mol

        except Exception as e:
            if self.show_progress:
                print(f"Error loading supermolecule from {path}: {e}")
            return None

    def _calculate_shape_score(self, query_mol: Chem.Mol) -> float:
        """
        Calculate the best shape score between a query molecule and the supermolecule.

        Tries all conformer combinations and returns the maximum score.

        Args:
            query_mol: Query molecule with conformers

        Returns:
            Maximum score across all conformer combinations
        """
        if query_mol is None or self.reference_mol is None:
            return 0.0

        if query_mol.GetNumConformers() == 0:
            return 0.0

        best_score = 0.0

        # Try all conformer combinations
        for query_conf in query_mol.GetConformers():
            for ref_conf in self.reference_mol.GetConformers():
                try:
                    # Make a copy of the query molecule (probe is modified during alignment)
                    probe_copy = Chem.Mol(query_mol)

                    # Perform alignment and get shape and color scores
                    result = rdShapeAlign.AlignMol(
                        self.reference_mol,
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
        """
        Score molecules against the supermolecule reference.

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
        import os
        import tempfile
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
            from collections import defaultdict
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

                    # Calculate maximum score across all conformers
                    max_score = 0.0
                    for conf_mol in mol_conformers:
                        score = self._calculate_shape_score(conf_mol)
                        max_score = max(max_score, score)

                    scores[mol_id] = max_score

                    # Progress indication
                    if self.show_progress and (mol_id + 1) % 100 == 0:
                        print(f"  Scored {mol_id + 1}/{num_mols} molecules")
            else:
                # Parallel processing
                # Prepare arguments for workers
                worker_args = [
                    (mol_id, conformers_by_mol.get(mol_id, []), self.reference_mol,
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
                                    pool.imap(_score_molecule_supermol_worker, worker_args, chunksize=chunksize),
                                    total=num_mols,
                                    desc="Scoring molecules"
                                ))
                            except ImportError:
                                # Fallback without progress bar
                                results = pool.map(_score_molecule_supermol_worker, worker_args, chunksize=chunksize)
                                print(f"  Scored {num_mols} molecules (parallel)")
                        else:
                            results = pool.map(_score_molecule_supermol_worker, worker_args, chunksize=chunksize)

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
                            score = self._calculate_shape_score(conf_mol)
                            max_score = max(max_score, score)
                        scores[mol_id] = max_score

        if self.show_progress:
            print(f"Scoring complete. Average score: {scores.mean():.3f}")

        return scores
