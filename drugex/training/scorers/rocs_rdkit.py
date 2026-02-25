"""RDKit-based ROCS scorer."""
import os
import tempfile
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdShapeAlign, AllChem

from drugex.training.scorers.interfaces import ConformerGenerator, Scorer

_RDKIT_WORKER_SETTINGS: Dict[str, object] = {}
_DEFAULT_RDKIT_GROUP_NAME = "_default_group"


def _score_single_reference(
    query_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    score_type: str,
    use_colors: bool,
) -> float:
    """Compute best alignment score between a query molecule and one reference.

    Uses rdShapeAlign.AlignMol which performs Gaussian shape overlay (same
    algorithm family as OpenEye ROCS and CDPKit GaussianShapeAlignment).

    Note on opt_param: AlignMol defaults to opt_param=1.0 (shape-only
    optimization). CDPKit defaults to TotalOverlapTanimoto. Neither directly
    optimizes TanimotoCombo during alignment. In theory this means the color
    component is evaluated at a shape-optimized pose rather than jointly
    optimized. In practice, benchmarking with CCR2-like compounds shows the
    max TanimotoCombo is identical across opt_param=0.0/0.5/1.0 when enough
    conformer pairs are sampled (200 conformers x 5 references), so the
    choice of optimization objective has no effect on final scores.
    """
    if query_mol is None or ref_mol is None:
        return 0.0
    if query_mol.GetNumConformers() == 0 or ref_mol.GetNumConformers() == 0:
        return 0.0

    best_score = 0.0
    for query_conf in query_mol.GetConformers():
        for ref_conf in ref_mol.GetConformers():
            try:
                probe_copy = Chem.Mol(query_mol)
                result = rdShapeAlign.AlignMol(
                    ref_mol,
                    probe_copy,
                    refConfId=ref_conf.GetId(),
                    probeConfId=query_conf.GetId(),
                    useColors=use_colors,
                )
            except (RuntimeError, ValueError):
                continue

            if not isinstance(result, (list, tuple)) or len(result) < 2:
                continue

            shape_score, color_score = result[0], result[1]
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
) -> None:
    """Initializer to share immutable worker state."""
    global _RDKIT_WORKER_SETTINGS
    _RDKIT_WORKER_SETTINGS = {
        "reference_mols": reference_mols,
        "group_to_indices": group_to_indices,
        "score_type": score_type,
        "use_colors": use_colors,
    }


def _score_molecule_rdkit_worker(args: Tuple[int, List[Chem.Mol]]) -> Tuple[int, List[float]]:
    """Score a molecule in a worker process."""
    mol_id, mol_conformers = args
    settings = _RDKIT_WORKER_SETTINGS
    reference_mols: List[Chem.Mol] = settings.get("reference_mols", [])
    group_to_indices: List[List[int]] = settings.get("group_to_indices", [])
    score_type: str = settings.get("score_type", "TanimotoCombo")
    use_colors: bool = settings.get("use_colors", True)
    num_groups = len(group_to_indices) if group_to_indices else (1 if reference_mols else 0)
    if not mol_conformers or num_groups == 0:
        return mol_id, [0.0] * num_groups

    group_scores = [0.0] * num_groups
    try:
        for conf_mol in mol_conformers:
            if conf_mol is None or conf_mol.GetNumConformers() == 0:
                continue
            for group_idx, ref_indices in enumerate(group_to_indices):
                for ref_idx in ref_indices:
                    ref_mol = reference_mols[ref_idx]
                    score = _score_single_reference(conf_mol, ref_mol, score_type, use_colors)
                    if score > group_scores[group_idx]:
                        group_scores[group_idx] = score
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

        Raises:
            TypeError: If reference inputs use unsupported types.
            FileNotFoundError: If a reference path is missing.
            ValueError: If reference molecules lack conformers or cannot load.
        """
        super().__init__()

        self.conformer_generator = conformer_generator
        self.score_type = score_type
        self.use_colors = use_colors
        self.show_progress = show_progress
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

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

    def getKey(self) -> List[str]:
        if (
            len(self.group_names) == 1
            and self.group_names[0] == _DEFAULT_RDKIT_GROUP_NAME
        ):
            prefix = "RDKit_Supermol" if self._single_reference else "RDKit_Aggregate"
            refs = len(self.reference_mols)
            if prefix == "RDKit_Aggregate":
                return [f"{prefix}_{refs}refs_{self.score_type}"]
            return [f"{prefix}_{self.score_type}"]
        return [f"RDKit_{name}" for name in self.group_names]

    def _calculate_shape_score(self, query_mol: Chem.Mol, ref_mol: Chem.Mol) -> float:
        return _score_single_reference(query_mol, ref_mol, self.score_type, self.use_colors)

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
                            conf_mol, ref_mol, self.score_type, self.use_colors
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

        with tempfile.TemporaryDirectory() as tmpdir:
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
                    with Pool(
                        self.n_jobs,
                        initializer=_rdkit_worker_init,
                        initargs=(
                            self.reference_mols,
                            self.group_to_indices,
                            self.score_type,
                            self.use_colors,
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
