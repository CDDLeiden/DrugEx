import gc
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from rdkit import Chem

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from openeye import oechem, oeomega, oeshape

    OE_AVAILABLE = True
except ImportError:
    OE_AVAILABLE = False

from drugex.training.scorers.interfaces import Scorer


@dataclass
class ROCSPerformanceConfig:
    """Configuration for ROCS performance tuning"""

    memory_pressure_threshold: float = 0.8  # 0.8 %


# Global configuration instance
PERF_CONFIG = ROCSPerformanceConfig()


class MemoryManager:
    """Advanced memory monitoring and management"""

    @staticmethod
    def get_memory_info() -> str | Dict[str, float]:
        """Get current memory usage information"""
        if not PSUTIL_AVAILABLE:
            return "Unable to retrieve memory info, psutil not installed"

        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "used_percent": memory.percent,
            "free_gb": memory.free / (1024**3),
        }

    @staticmethod
    def check_memory_pressure() -> bool:
        """Check if memory pressure is high"""
        info = MemoryManager.get_memory_info()

        if not PSUTIL_AVAILABLE:
            return False

        return info["used_percent"] > PERF_CONFIG.memory_pressure_threshold

    @staticmethod
    def force_cleanup():
        """Aggressive memory cleanup"""
        gc.collect()
        try:
            if hasattr(oechem, "OEClearMemory"):
                oechem.OEClearMemory()  # FIXME: This does not exist!?
        except:
            pass

    @staticmethod
    def log_memory_usage(context: str = ""):
        """Log current memory usage (silent in production)"""
        print(f"Memory usage {context}: {MemoryManager.get_memory_info()}")


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


class CLIROCSScorer(Scorer):
    """CLI ROCS scorer with multi-query support

    Features:
    - Multiple .sq query file support
    - Best score selection across queries
    - RDKit molecule support
    - Batch processing for large datasets

    Attributes:
        - query_files: List of .sq files for ROCS queries
        - score_type: Type of scoring to use (e.g., TanimotoCombo)
        - max_conformers: Maximum conformers per molecule (note. max 200)
        - max_isomers: Maximum isomers per molecule
        - max_heavy_atoms: Maximum heavy atoms per molecule
        - max_rotatable_bonds: Maximum rotatable bonds per molecule
        - shape_only: If True, only shape scoring is performed
        - optimize: If True,
        - color_optimize: If True, color optimization is performed
        - color_force_field: Force field to use for color optimization
        - use_gpu: If True, GPU acceleration is used (if available)
        - rocs_binary: Name of the ROCS binary to use
        - binary_path: Path to the ROCS binary (if not in PATH)
        - output_file: Output file for ROCS results
        - show_progress: If True, progress is shown during scoring
        - name_suffix: Optional suffix for the scorer name (important if multiple
           CLIROCSScorer scorers are used in the same environment)
    """

    def __init__(
        self,
        query_files: Union[str, List[str]],
        score_type: str = "TanimotoCombo",
        max_conformers: int = 10,
        max_isomers: int = 4,
        max_heavy_atoms: int = 35,
        max_rotatable_bonds: int = 15,
        shape_only: bool = False,
        optimize: bool = True,
        color_optimize: bool = True,
        color_force_field: str = "ImplicitMillsDean",
        use_gpu: bool = False,
        rocs_binary: str = "rocs",
        binary_path: str | None = None,
        output_file: str | None = None,
        show_progress: bool = False,
        name_suffix: str | None = None,
    ):

        super().__init__()

        if not OE_AVAILABLE:
            raise ImportError("OpenEye toolkits required")

        # Convert to list and validate
        self.query_files = (
            [query_files] if isinstance(query_files, str) else list(query_files)
        )

        self.score_type = score_type
        self.optimize = optimize
        self.color_optimize = color_optimize
        self.color_force_field = color_force_field
        self.binary_path = binary_path or rocs_binary
        self.output_file = output_file

        self.max_conformers = max_conformers
        if self.max_conformers > 200:
            print(
                "Warning: max_conformers > 200 may cause memory issues "
                "Setting to 200."
            )
            self.max_conformers = 200
        self.max_isomers = max_isomers
        self.max_heavy_atoms = max_heavy_atoms
        self.max_rotatable_bonds = max_rotatable_bonds

        self.use_gpu = use_gpu
        self.shape_only = shape_only
        self.rocs_binary = rocs_binary
        self.show_progress = show_progress
        self.name_suffix = name_suffix
        self._validate_query_files()

        if not shutil.which(self.binary_path):
            raise FileNotFoundError(f"ROCS binary not found: {self.binary_path}")

    def _validate_query_files(self):
        """Validate all .sq files exist and are readable"""
        valid_files = []
        for qf in self.query_files:
            if not os.path.exists(qf):
                print(f"Query file not found: {qf}")
                continue

            try:
                query = oeshape.OEShapeQuery()
                if oeshape.OEReadShapeQuery(qf, query):
                    valid_files.append(os.path.abspath(qf))
            except Exception:
                continue

        if not valid_files:
            raise ValueError("No valid query files found")
        self.query_files = valid_files

    def _create_fresh_omega(self):
        """Create a fresh Omega instance with proven parameters"""
        opts = oeomega.OEOmegaOptions()
        # Use conservative conformer limits
        opts.SetMaxConfs(self.max_conformers)
        opts.SetStrictStereo(False)
        opts.SetFromCT(True)
        opts.SetFixRMS(True)
        opts.SetRMSThreshold(0.5)
        opts.SetEnumRing(True)
        opts.SetRotorOffset(False)

        # Force CPU mode to reduce memory pressure
        opts.GetTorDriveOptions().SetUseGPU(False)
        opts.SetSampleHydrogens(True)

        return oeomega.OEOmega(opts)

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

    def getScores(self, mols, frags=None) -> np.ndarray:
        """Score molecules using one or more ROCS queries"""
        if not mols:
            print("No molecules to score")
            return np.zeros(0)

        timer = oechem.OEWallTimer() if self.show_progress else None
        num_input_mols = len(mols)

        if self.show_progress:
            print(f"Starting ROCS scoring for {num_input_mols} molecules...")
            MemoryManager.log_memory_usage("before scoring")

        # Convert to SMILES list for uniform processing
        smiles_list = self._convert_to_smiles(mols)

        # Prepare conformers
        with _managed_tmpdir() as tmpdir:
            conf_file = self._prepare_molecules_for_cli(smiles_list, tmpdir)

            # Score using OpenEye ROCS
            scores_dict = self._score(conf_file)

        result_scores = np.zeros(num_input_mols, dtype=np.float32)
        result_scores[list(scores_dict.keys())] = list(scores_dict.values())

        if self.show_progress:
            if timer and timer.Elapsed() > 2.0:
                print(f"ROCS scoring completed in {timer.Elapsed():.1f}s")
            MemoryManager.log_memory_usage("after scoring")

        return result_scores

    def _filter_mol(self, smi, mol) -> bool:
        """Filter molecules based on heavy atoms and rotatable bonds"""

        # filter based on heavy atoms and rotatable bonds
        if oechem.OECount(mol, oechem.OEIsHeavy()) > self.max_heavy_atoms:
            oechem.OEThrow.Warning(
                f"Skipping {smi} with > {self.max_heavy_atoms} heavy atoms"
            )
            return True

        if oechem.OECount(mol, oechem.OEIsRotor()) > self.max_rotatable_bonds:
            oechem.OEThrow.Warning(
                f"Skipping {smi} with > {self.max_rotatable_bonds} rotatable bonds"
            )
            return True
        return False

    def _get_isomers(self, mol):
        """Generate isomers for a molecule using OMEGA"""
        opts = oeomega.OEFlipperOptions()
        opts.SetMaxCenters(self.max_isomers)
        for conf in oeomega.OEFlipper(mol, opts):
            iso = oechem.OEMol(conf)
            yield iso

    def _prepare_molecules_for_cli(self, smiles_list, temp_dir) -> str:
        """Prepare molecules with conformers for CLI scoring"""
        output_file = os.path.join(temp_dir, f"conformers.oeb.gz")
        ofs = oechem.oemolostream()
        if not ofs.open(output_file):
            oechem.OEThrow.Fatal(
                "Unable to open %s for writing conformers" % output_file
            )

        omega = self._create_fresh_omega()

        # Progress tracking for conformer generation
        dots = None
        if self.show_progress and len(smiles_list) > 50:
            print("Generating conformers...")
            dots = oechem.OEThreadedDots(100, 50, "molecules")

        for i, smi in enumerate(smiles_list):
            mol = oechem.OEMol()
            title = f"mol_{i}"
            mol.SetTitle(title)

            if smi is None or not oechem.OESmilesToMol(mol, smi):
                continue

            for j, iso in enumerate(self._get_isomers(mol)):
                iso.SetTitle(f"{title}+{j}")
                ret_code = omega.Build(iso)
                if ret_code == oeomega.OEOmegaReturnCode_Success:
                    oechem.OEWriteMolecule(ofs, iso)
                else:
                    oechem.OEThrow.Warning(
                        "%s: %s %s"
                        % (smi, iso.GetTitle(), oeomega.OEGetOmegaError(ret_code))
                    )

                if dots:
                    dots.Update()

        if dots:
            dots.Total()

        ofs.close()
        omega = None
        MemoryManager.force_cleanup()

        return output_file

    def _build_rocs_command(
        self, query_file: str, input_file: str, output_file: str
    ) -> List[str]:
        """Build ROCS command matching cli_base_rocs.py exactly"""
        # Extract the directory from output_file
        output_dir = os.path.dirname(output_file) or "."

        cmd = [
            self.binary_path,
            "-query",
            query_file,
            "-dbase",
            input_file,
            "-report",
            "one",
            "-reportfile",
            output_file,
            "-prefix",
            "rocs",
            "-outputdir",
            output_dir,  # Add output directory
            "-maxconfs",
            str(self.max_conformers),
            "-rankby",
            self.score_type,
            "-chemff",
            self.color_force_field,
            "-cutoff",
            "-1.0",  # Return all molecules (no cutoff)
            "-maxhits",
            "0",  # Return all molecules (overrides besthits)
            "-tanimoto_cutoff",
            "0.0",
            "-stats",
            "best",
            "-nostructs",
        ]

        # Add shapeonly explicitly (with true/false value) instead of conditionally
        cmd.extend(["-shapeonly", str(self.shape_only).lower()])

        # Add opt explicitly instead of conditionally
        cmd.extend(["-opt", str(self.optimize).lower()])

        # Add optchem explicitly with proper condition
        if not self.shape_only and self.color_optimize:
            cmd.extend(["-optchem", "true"])
        else:
            cmd.extend(["-optchem", "false"])

        return cmd

    def _score(self, conf_file) -> dict:
        """Score molecules with ROCS"""
        # Check memory before each attempt
        if MemoryManager.check_memory_pressure():
            MemoryManager.force_cleanup()
            time.sleep(1)  # Brief pause for system recovery

        # Multi-query scoring
        if len(self.query_files) == 1:
            scores_dict = self._score_single_query(conf_file, self.query_files[0])
        else:
            scores_dict = self._score_multi_query(conf_file)

        return scores_dict

    def _score_multi_query(self, conf_file) -> dict:
        """Score against multiple queries, return best scores as dict"""
        best_scores = {}

        for query_file in self.query_files:
            query_scores = self._score_single_query(conf_file, query_file)

            # Take maximum score for each molecule
            for mol_title, score in query_scores.items():
                best_scores[mol_title] = max(score, best_scores.get(mol_title, 0.0))

        return best_scores

    def _write_molecules(self, mols, tmpdir) -> str:
        """Write molecules to OEB file"""
        input_file = os.path.join(tmpdir, "input.oeb")
        ofs = oechem.oemolostream()

        if not ofs.open(input_file):
            raise IOError(f"Cannot create input file: {input_file}")

        written_count = 0
        for mol in mols:
            if mol and mol.NumAtoms() > 0:
                oechem.OEWriteMolecule(ofs, mol)
                written_count += 1
        ofs.close()

        if written_count == 0:
            raise RuntimeError("No valid molecules were written to input file")

        return input_file

    def _score_single_query(self, conf_file, query_file: str) -> dict:
        """Score molecules against a single query file and return as dict"""
        scores = {}

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
                timeout=300,
                env=dict(os.environ, OMP_NUM_THREADS="1"),
            )

            if self.show_progress and rocs_timer and rocs_timer.Elapsed() > 2.0:
                print(f"  ROCS execution: {rocs_timer.Elapsed():.1f}s")

            if result.returncode != 0:
                raise RuntimeError(f"ROCS failed with return code {result.returncode}")

            if not os.path.exists(output_file):
                raise RuntimeError(f"ROCS output file not created: {output_file}")

            if os.path.getsize(output_file) == 0:
                raise RuntimeError(f"ROCS output file is empty: {output_file}")

        except RuntimeError as e:
            raise RuntimeError(e)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ROCS execution timed out")
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

    def getKey(self) -> str:
        """Return scorer identifier"""
        if self.name_suffix:
            return f"ROCS_{self.name_suffix}"
        else:
            return "ROCS"
