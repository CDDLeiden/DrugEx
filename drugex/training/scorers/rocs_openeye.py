"""OpenEye-ROCS based ROCS scorer."""
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem

try:
    from openeye import oechem, oeshape

    OE_AVAILABLE = True
except ImportError:
    OE_AVAILABLE = False

from drugex.training.scorers.interfaces import ConformerGenerator, Scorer


@contextmanager
def _managed_tmpdir():
    """Managed temporary directory with guaranteed cleanup."""
    path = tempfile.mkdtemp(prefix="cli_rocs_")
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            print(f"Error cleaning up temporary directory {path}: {e}")


class OpenEyeROCSScorer(Scorer):
    """OpenEye ROCS scorer that shells out to the ROCS command-line binary.

    Uses the OpenEye ROCS CLI via ``subprocess`` for shape-based similarity
    scoring while relying on the OpenEye Python toolkits for reference validation
    and RDKit for molecule handling. Requires a valid OpenEye license with the
    ROCS binary available in ``PATH`` (or provided via ``binary_path``). If a
    future fastROCS implementation is added, this class will remain the CLI
    variant.

    Features:
    - Multiple reference group support (.sq or molecule files)
    - Best score selection across reference groups
    - Hybrid Python/CLI workflow for flexible integration

    Attributes:
        - conformer_generator: conformer generator used for generated molecules.
        - references: dict of reference files for ROCS scoring (.sq or molecule files).
            Dictionary keys are reference group names, values are file paths or lists
            of file paths. For each key (group), one score is returned per molecule.
            If a list of files is provided for a single group, the highest score across
            all reference files in that group is returned.
        - score_type: Type of scoring to use (e.g., TanimotoCombo)
        - shape_only: If True, only shape scoring is performed
        - optimize: If True, optimization is performed during scoring
        - color_optimize: If True, color optimization is performed
        - color_force_field: Force field to use for color optimization
        - rocs_binary: Name of the ROCS binary to use
        - binary_path: Path to the ROCS binary (if not in PATH)
        - show_progress: If True, progress is shown during scoring
        - name_suffix: Optional suffix for the scorer name (important if multiple
           OpenEyeROCSScorer scorers are used in the same environment)
    """

    def __init__(
        self,
        conformer_generator: ConformerGenerator,
        references: dict[str, List[str] | str],
        score_type: str = "TanimotoCombo",
        shape_only: bool = False,
        optimize: bool = True,
        color_optimize: bool = True,
        color_force_field: str = "ImplicitMillsDean",
        rocs_binary: str = "rocs",
        binary_path: str | None = None,
        show_progress: bool = True,
        timeout: int = 300,
    ):
        """Initialize the OpenEye ROCS scorer.

        Args:
            conformer_generator: generator used to produce generated molecule conformers
            references: dict mapping reference group names to SDF/sq reference file path(s).
                If multiple dict items are specified, the output will be a score per group.
                If a dict item has multiple file paths specified, molecules will be scored
                against each reference file but only the maximum score will be returned
                (i.e. the best match within that group).
            score_type: Type of scoring to use (e.g., TanimotoCombo), ignored
                if shape_only
            shape_only: If True, only shape scoring is performed
            optimize: turn optimizer on/off, if off score only
            color_optimize: If True, color optimization is performed
            color_force_field: Force field to use for color optimization
            rocs_binary: Name of the ROCS binary to use
            binary_path: Path to the ROCS binary (if not in PATH)
            show_progress: If True, progress is shown during scoring
            timeout: Timeout in seconds for the ROCS subprocess

        Raises:
            ImportError: If OpenEye toolkits are not available.
            FileNotFoundError: If rocs binary not found
        """
        super().__init__()

        if not OE_AVAILABLE:
            raise ImportError("OpenEye toolkits required")

        self.conformer_generator = conformer_generator

        # Convert to list and validate
        self.queries = references
        self._validate_query_files()

        self.score_type = score_type
        self.optimize = optimize
        self.color_optimize = color_optimize
        self.color_force_field = color_force_field
        self.binary_path = binary_path or rocs_binary
        self.timeout = timeout

        self.shape_only = shape_only
        self.rocs_binary = rocs_binary
        self.show_progress = show_progress

        if not shutil.which(self.binary_path):
            raise FileNotFoundError(f"ROCS binary not found: {self.binary_path}")

    def _validate_query_files(self):
        """Validate all reference files exist and are readable"""
        assert isinstance(self.queries, dict), (
            "references must be a dictionary with keys as reference group names and values "
            "as file paths"
        )

        for name, list_of_qf in self.queries.items():
            if isinstance(list_of_qf, str):
                list_of_qf = [list_of_qf]
                self.queries[name] = list_of_qf

            for qf in list_of_qf:
                if not os.path.exists(qf):
                    raise FileNotFoundError(f"Reference file not found: {qf}")
                ext = oechem.OEGetFileExtension(qf)
                if ext == "sq":
                    query = oeshape.OEShapeQuery()
                    if not oeshape.OEReadShapeQuery(qf, query):
                        raise ValueError(f"Invalid reference file: {qf}")
                else:
                    qfs = oechem.oemolistream()
                    if not qfs.open(qf):
                        oechem.OEThrow.Fatal("Unable to open '%s'" % qf)
                    query = oechem.OEGraphMol()
                    if not oechem.OEReadMolecule(qfs, query):
                        oechem.OEThrow.Fatal("Unable to read query from '%s'" % qf)

    def _convert_to_smiles(self, mols) -> List[str | None]:
        """Convert various molecule types to SMILES"""
        smiles_list = []
        for mol in mols:
            if mol is None:
                smiles_list.append(None)
            elif isinstance(mol, str):
                smiles_list.append(mol)
            elif hasattr(mol, "GetTitle"):  # OpenEye molecule
                smi = oechem.OECreateSmiString(mol)
                smiles_list.append(smi)
            elif hasattr(mol, "GetNumAtoms"):  # RDKit molecule
                smiles_list.append(Chem.MolToSmiles(mol))
            else:
                raise TypeError(f"Unsupported molecule type: {type(mol)}")
        return smiles_list

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

    def getScores(self, mols, frags=None) -> np.ndarray:
        """Score molecules using one or more ROCS queries"""
        if not mols:
            print("No molecules to score")
            return np.zeros(0)

        timer = oechem.OEWallTimer() if self.show_progress else None
        num_input_mols = len(mols)

        if self.show_progress:
            print(f"Starting ROCS scoring for {num_input_mols} molecules...")

        # Convert to SMILES list for uniform processing
        smiles_list = self._convert_to_smiles(mols)

        # Deduplicate SMILES to avoid redundant conformer generation
        unique_smiles, unique_to_original = self._deduplicate_smiles(smiles_list)

        if not unique_smiles:
            return np.zeros((num_input_mols, len(self.queries)), dtype=np.float32)

        if self.show_progress and len(unique_smiles) < len(smiles_list):
            print(
                f"  Deduplicated: {len(smiles_list)} -> {len(unique_smiles)} "
                f"unique SMILES"
            )

        # Prepare conformers for unique SMILES only
        with _managed_tmpdir() as tmpdir:
            conf_file = self.conformer_generator.genConformers(
                unique_smiles, tmpdir
            )

            # Score using OpenEye ROCS
            scores_dict = self._score(conf_file)

        # Map scores from unique indices back to original indices
        result_scores = np.zeros((num_input_mols, len(self.queries)), dtype=np.float32)
        for i, scores in enumerate(scores_dict.values()):
            for unique_idx, score in scores.items():
                if unique_idx not in unique_to_original:
                    continue
                for original_idx in unique_to_original[unique_idx]:
                    result_scores[original_idx, i] = score

        if self.show_progress:
            if timer and timer.Elapsed() > 2.0:
                print(f"ROCS scoring completed in {timer.Elapsed():.1f}s")

        return result_scores

    def _build_rocs_command(
        self, query_file: str, input_file: str, output_file: str
    ) -> List[str]:
        """Build ROCS command matching cli_base_rocs.py exactly"""
        # Extract the directory from output_file
        output_dir = os.path.dirname(output_file) or "."

        cmd = [
            self.binary_path,
            "-query", query_file,
            "-dbase", input_file,
            "-report", "one",
            "-reportfile", output_file,
            "-prefix", "rocs",
            "-outputdir", output_dir,
            "-stats", "best", # Include best overlay(s) for each dbase molecule
            "-nostructs", # Don't save the structures
            "-scdbase",  # Don't combine contiguous conformers
        ]
        
        if self.shape_only:
            # sets chemff none, optchem false and rankby tanimoto
            cmd.extend(["-shapeonly", "true"])
        else:
            cmd.extend(["-rankby", self.score_type])
            cmd.extend(["-chemff", self.color_force_field])

        # set optimizer on/off, if off score only
        cmd.extend(["-opt", str(self.optimize).lower()])

        # wire color_optimize to -optchem (only meaningful with optimizer on
        # and shape_only off — ROCS ignores it otherwise)
        if not self.shape_only and self.optimize:
            cmd.extend(["-optchem", str(self.color_optimize).lower()])

        return cmd

    def _score(self, conf_file) -> dict:
        """Score molecules with ROCS

        Returns:
            dict: Dictionary with reference group names as keys and scores as values.
        """
        # Multi-group scoring
        scores_dict = {}
        for name, query_files in self.queries.items():
            if len(query_files) == 1:
                scores_dict[name] = self._score_single_query(conf_file, query_files[0])
            else:
                scores_dict[name] = self._score_multi_query(conf_file, query_files)

        return scores_dict

    def _score_multi_query(self, conf_file, query_files) -> dict:
        """Score against multiple reference files in a group, return best scores as dict"""
        best_scores = {}

        for query_file in query_files:
            query_scores = self._score_single_query(conf_file, query_file)

            # Take maximum score for each molecule across all reference files
            for mol_title, score in query_scores.items():
                best_scores[mol_title] = max(score, best_scores.get(mol_title, 0.0))

        return best_scores

    def _score_single_query(self, conf_file, query_file: str) -> dict:
        """Score molecules against a single reference file and return as dict"""
        scores = {}
        print("Scoring with reference file:", query_file)

        with _managed_tmpdir() as tmpdir:
            if not conf_file:
                print("No valid molecules written to input file")
                return scores
            # Execute ROCS
            output_file = self._execute_rocs(query_file, conf_file, tmpdir)
            if not output_file:
                print("ROCS execution failed or output file not created")
                return scores

            scores = self._parse_results(output_file)

        return scores

    def _execute_rocs(self, query_file: str, input_file: str, tmpdir: str) -> str:
        """Execute ROCS CLI"""
        output_file = os.path.join(tmpdir, "rocs_output.tsv")

        cmd = self._build_rocs_command(query_file, input_file, output_file)

        try:
            # Check prerequisites
            if not os.path.exists(query_file):
                raise RuntimeError(f"Query file not found: {query_file}")
            if not os.path.exists(input_file):
                raise RuntimeError(f"Input file not found: {input_file}")
            if not shutil.which(self.binary_path):
                raise RuntimeError(f"ROCS binary not found: {self.binary_path}")

            # Execute ROCS with timing
            rocs_timer = oechem.OEWallTimer() if self.show_progress else None
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if self.show_progress and rocs_timer and rocs_timer.Elapsed() > 2.0:
                print(f"  ROCS execution: {rocs_timer.Elapsed():.1f}s")

            if result.returncode != 0:
                raise RuntimeError(
                    f"ROCS failed with return code {result.returncode}:\n{result.stderr}"
                )

            if not os.path.exists(output_file):
                raise RuntimeError(f"ROCS output file not created: {output_file}")

            if os.path.getsize(output_file) == 0:
                raise RuntimeError(f"ROCS output file is empty: {output_file}")

        except RuntimeError as e:
            raise RuntimeError(e)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"ROCS execution timed out after {self.timeout}s"
            )
        except Exception as e:
            raise RuntimeError(f"ROCS execution failed: {e}")

        return output_file

    def _parse_results(self, output_file: str) -> dict[str, float]:
        """Parse ROCS output and return scores as a dictionary"""
        scores = {}

        if not os.path.exists(output_file):
            print(f"Output file not found: {output_file}")
            return scores

        df = pd.read_csv(output_file, sep="\t")

        if df.empty:
            print("ROCS output file is empty")
            return scores

        if "Name" not in df.columns or self.score_type not in df.columns:
            print("ROCS output file is missing required columns")
            return scores

        # find the maximum score per molecule out of all its conformers/isomers
        for _, row in df.iterrows():
            # Extract molecule ID from "mol_<id>+<conf>"
            try:
                mol_id = int(row["Name"].split("+")[0].split("_")[1])
            except IndexError:
                raise ValueError(f"Invalid molecule name format: {row['Name']}")
            conf_score = float(row[self.score_type])
            mol_max_score = scores.get(mol_id, 0.0)
            scores[mol_id] = max(mol_max_score, conf_score)

        return scores

    def getKey(self) -> List[str]:
        """Return scorer identifier"""
        return [f"ROCS_{name}" for name in self.queries.keys()]
