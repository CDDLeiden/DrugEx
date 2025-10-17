"""RDKit-based ROCS scorer."""

import os
import tempfile
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import List, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdShapeAlign

from drugex.training.scorers.interfaces import ConformerGenerator, Scorer


def _score_molecule_rdkit_worker(args):
    """Score a molecule in a worker process."""
    mol_id, mol_conformers, reference_mols, score_type, use_colors = args
    if not mol_conformers:
        return mol_id, 0.0

    max_score = 0.0
    try:
        for conf_mol in mol_conformers:
            if conf_mol is None or conf_mol.GetNumConformers() == 0:
                continue
            for ref_mol in reference_mols:
                if ref_mol is None or ref_mol.GetNumConformers() == 0:
                    continue
                for query_conf in conf_mol.GetConformers():
                    for ref_conf in ref_mol.GetConformers():
                        try:
                            probe_copy = Chem.Mol(conf_mol)
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
                        max_score = max(max_score, score)
    except Exception:
        return mol_id, 0.0

    return mol_id, max_score


class RDKitROCSScorer(Scorer):
    """ROCS-style scoring using RDKit shape alignment.

    Features:
    - Accepts references as files or RDKit molecules with conformers.
    - Supports shape-only, color-only, and combo scoring modes.
    - Optional multiprocessing and verbose progress reporting.

    Attributes:
        conformer_generator: Generator used to build query conformers.
        reference_mols: Normalized list of reference molecules.
        score_type: Requested score mode (`shape`, `color`, or `TanimotoCombo`).
        use_colors: Whether to include pharmacophore colors in alignment.
        show_progress: Enables logging for long runs.
        n_jobs: Requested worker count (-1 maps to available CPUs).
        _single_reference: True when operating in single-reference mode.
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
        """Initialize the RDKit ROCS scorer.

        Args:
            conformer_generator: Conformer generator used for query molecules.
            references: Reference source as SDF path(s) or RDKit molecule(s).
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
        self._single_reference = self._is_single_reference(references)
        self.reference_mols = self._normalize_references(references)
        self._validate_references()

    @staticmethod
    def _is_single_reference(
        references: Union[str, List[str], Chem.Mol, List[Chem.Mol]]
    ) -> bool:
        if isinstance(references, (str, Chem.Mol)):
            return True
        if isinstance(references, list) and len(references) == 1:
            return True
        return False

    def _normalize_references(
        self, references: Union[str, List[str], Chem.Mol, List[Chem.Mol]]
    ) -> List[Chem.Mol]:
        if isinstance(references, str):
            return [self._load_molecule_from_file(references)]

        if isinstance(references, list) and all(isinstance(r, str) for r in references):
            return [self._load_molecule_from_file(path) for path in references]

        if isinstance(references, Chem.Mol):
            return [references]

        if isinstance(references, list) and all(isinstance(r, Chem.Mol) for r in references):
            return references

        raise TypeError(
            "references must be str, List[str], Chem.Mol, or List[Chem.Mol]"
        )

    def _load_molecule_from_file(self, path: str) -> Chem.Mol:
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
        except Exception as exc:
            if self.show_progress:
                print(f"Warning: failed to load {path}: {exc}")
            raise ValueError(f"Failed to load molecule from {path}: {exc}") from exc

    def _validate_references(self):
        if not self.reference_mols:
            raise ValueError("At least one reference molecule is required")

        for idx, ref_mol in enumerate(self.reference_mols):
            if ref_mol is None:
                raise ValueError(f"Reference molecule at index {idx} is None")
            if ref_mol.GetNumConformers() == 0:
                raise ValueError(
                    f"Reference molecule at index {idx} has no conformers."
                )

    def getKey(self) -> List[str]:
        prefix = "RDKit_Supermol" if self._single_reference else "RDKit_Aggregate"
        refs = len(self.reference_mols)
        if prefix == "RDKit_Aggregate":
            return [f"{prefix}_{refs}refs_{self.score_type}"]
        return [f"{prefix}_{self.score_type}"]

    def _calculate_shape_score(self, query_mol: Chem.Mol, ref_mol: Chem.Mol) -> float:
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
                        useColors=self.use_colors,
                    )
                except (RuntimeError, ValueError) as exc:
                    if self.show_progress:
                        print(f"Warning: shape alignment failed: {exc}")
                    continue

                if not isinstance(result, (list, tuple)) or len(result) < 2:
                    continue

                shape_score, color_score = result[0], result[1]
                if self.score_type == "shape":
                    score = shape_score
                elif self.score_type == "color":
                    score = color_score
                else:
                    score = shape_score + color_score

                best_score = max(best_score, score)

        return best_score

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

    def getScores(self, mols: List[Chem.Mol], frags=None) -> np.ndarray:
        if not mols:
            return np.zeros((0, 1))

        num_mols = len(mols)
        scores = np.zeros((num_mols, 1))

        if self.show_progress:
            print(f"Scoring {num_mols} molecules with {self.getKey()}...")

        smiles_list = self._convert_to_smiles(mols)

        with tempfile.TemporaryDirectory() as tmpdir:
            conf_file = self.conformer_generator.genConformers(smiles_list, tmpdir)
            if not os.path.exists(conf_file):
                if self.show_progress:
                    print("Warning: conformer generation failed")
                return scores

            conformers_by_mol = defaultdict(list)
            try:
                suppl = Chem.SDMolSupplier(conf_file, removeHs=False)
                for conf_mol in suppl:
                    if conf_mol is None:
                        continue
                    try:
                        name = conf_mol.GetProp("_Name")
                        mol_id = int(name.split("+")[0].split("_")[1])
                        conformers_by_mol[mol_id].append(conf_mol)
                    except Exception:
                        continue
            except Exception:
                if self.show_progress:
                    print(f"Warning: failed to load conformers from {conf_file}")
                return scores

            if self.n_jobs == 1:
                for mol_id in range(num_mols):
                    mol_conformers = conformers_by_mol.get(mol_id, [])
                    if not mol_conformers:
                        continue
                    max_score = 0.0
                    for conf_mol in mol_conformers:
                        for ref_mol in self.reference_mols:
                            score = self._calculate_shape_score(conf_mol, ref_mol)
                            max_score = max(max_score, score)
                    scores[mol_id] = max_score
                    if self.show_progress and (mol_id + 1) % 100 == 0:
                        print(f"  Scored {mol_id + 1}/{num_mols} molecules")
            else:
                worker_args = [
                    (mol_id, conformers_by_mol.get(mol_id, []), self.reference_mols,
                     self.score_type, self.use_colors)
                    for mol_id in range(num_mols)
                ]
                effective_jobs = max(1, self.n_jobs)
                chunksize = max(1, num_mols // (effective_jobs * 4))

                try:
                    with Pool(self.n_jobs) as pool:
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
                                        total=num_mols,
                                        desc="Scoring molecules",
                                    )
                                )
                            except ImportError:
                                results = pool.map(
                                    _score_molecule_rdkit_worker,
                                    worker_args,
                                    chunksize=chunksize,
                                )
                                print(f"  Scored {num_mols} molecules (parallel)")
                        else:
                            results = pool.map(
                                _score_molecule_rdkit_worker,
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
                    for mol_id in range(num_mols):
                        mol_conformers = conformers_by_mol.get(mol_id, [])
                        if not mol_conformers:
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
