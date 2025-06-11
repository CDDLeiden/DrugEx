#!/usr/bin/env python3
"""
Smart CLI ROCS Scorer with Multi-Query Support

- Uses CLI ROCS for scoring
- Supports multiple .sq query files with best score selection
- Returns numpy arrays for DrugEx compatibility
"""

import os
import tempfile
import subprocess
import shutil
import numpy as np
from typing import Union, List

try:
    from openeye import oechem, oeomega, oeshape
    OE_AVAILABLE = True
except ImportError:
    OE_AVAILABLE = False

from drugex.training.scorers.interfaces import Scorer

class SmartCLIROCSScorer(Scorer):
    """
    Smart CLI ROCS scorer with multi-query support
    
    Features:
    - Multiple .sq query file support
    - Best score selection across queries
    - RDKit molecule support
    - Batch processing for large datasets
    """
    
    def __init__(self,
                 query_files: Union[str, List[str]],
                 max_conformers: int = 200,
                 max_isomers: int = 4,
                 max_heavy_atoms: int = 35,
                 max_rotatable_bonds: int = 15,
                 use_gpu: bool = False,
                 shape_only: bool = False,
                 rocs_binary: str = "rocs",
                 max_retry_attempts: int = 3,
                 batch_size_limit: int = 1000,
                 show_progress: bool = True):
        
        super().__init__()
        if not OE_AVAILABLE:
            raise ImportError("OpenEye toolkits required")
        
        # Convert to list and validate
        self.query_files = [query_files] if isinstance(query_files, str) else list(query_files)
        self.max_conformers = max_conformers
        self.max_isomers = max_isomers
        self.max_heavy_atoms = max_heavy_atoms
        self.max_rotatable_bonds = max_rotatable_bonds
        self.use_gpu = use_gpu
        self.shape_only = shape_only
        self.rocs_binary = rocs_binary
        self.max_retry_attempts = max_retry_attempts
        self.batch_size_limit = batch_size_limit
        self.show_progress = show_progress
        
        self._validate_query_files()
        
    def _validate_query_files(self):
        """Validate all .sq files exist and are readable"""
        valid_files = []
        for qf in self.query_files:
            if not os.path.exists(qf):
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
        
    def getScores(self, mols, frags=None) -> np.ndarray:
        """Score molecules against all query files, return best scores"""
        if not mols:
            return np.zeros(0)
            
        timer = oechem.OEWallTimer() if self.show_progress else None
        num_input_mols = len(mols)
        result_scores = np.zeros(num_input_mols)
        
        if self.show_progress:
            print(f"Starting ROCS scoring for {num_input_mols} molecules...")
        
        # Handle large batches by splitting them
        if num_input_mols > self.batch_size_limit:
            return self._process_large_batch(mols)
        
        try:
            # Convert to SMILES list for uniform processing
            smiles_list = self._convert_to_smiles(mols)
            
            # Process molecules for CLI
            processed_mols = self._prepare_molecules_for_cli(smiles_list)
            
            if not processed_mols:
                return result_scores
            
            # Multi-query scoring with retry logic
            scores_dict = self._score_with_retry(processed_mols)
                
            if not scores_dict:
                return result_scores
                
            # Map scores back to original indices
            for mol_key, score in scores_dict.items():
                try:
                    # Extract index from mol_X pattern
                    if mol_key.startswith('mol_'):
                        idx = int(mol_key.split('_')[1])
                        if 0 <= idx < num_input_mols:
                            result_scores[idx] = max(result_scores[idx], score)
                except (ValueError, IndexError):
                    continue
                
        except Exception:
            return result_scores
        
        if self.show_progress and timer and timer.Elapsed() > 2.0:
            print(f"ROCS scoring completed in {timer.Elapsed():.1f}s")
        
        return result_scores
    
    def _convert_to_smiles(self, mols) -> List[str]:
        """Convert various molecule types to SMILES"""
        smiles_list = []
        for mol in mols:
            if mol is None:
                smiles_list.append("")
            elif isinstance(mol, str):
                smiles_list.append(mol)
            elif hasattr(mol, 'GetTitle'):  # OpenEye molecule
                smi = oechem.OECreateSmiString(mol)
                smiles_list.append(smi if smi else "")
            elif hasattr(mol, 'GetNumAtoms'):  # RDKit molecule
                try:
                    from rdkit import Chem
                    smi = Chem.MolToSmiles(mol)
                    smiles_list.append(smi if smi else "")
                except:
                    smiles_list.append("")
            else:
                smiles_list.append("")
        return smiles_list
    
    def _prepare_molecules_for_cli(self, smiles_list) -> List[oechem.OEMol]:
        """Prepare molecules with conformers for CLI scoring"""
        processed = []
        omega = self._setup_omega()
        
        # Progress tracking for conformer generation
        valid_molecules = len([s for s in smiles_list if s.strip()])
        dots = None
        if self.show_progress and valid_molecules > 50:
            print("Generating conformers...")
            dots = oechem.OEThreadedDots(100, 50, "molecules")
        
        for i, smi in enumerate(smiles_list):
            if not smi or not smi.strip():
                continue
                
            try:
                mol = oechem.OEMol()
                if not oechem.OESmilesToMol(mol, smi.strip()):
                    continue
                    
                mol.SetTitle(f"mol_{i}")
                
                # Generate conformers
                molecule_processed = False
                try:
                    if omega(mol):
                        processed.append(mol)
                        molecule_processed = True
                    else:
                        # Fallback without conformers
                        mol_copy = oechem.OEMol()
                        if oechem.OESmilesToMol(mol_copy, smi.strip()):
                            mol_copy.SetTitle(f"mol_{i}")
                            processed.append(mol_copy)
                            molecule_processed = True
                except Exception:
                    # Final fallback
                    mol_copy = oechem.OEMol()
                    if oechem.OESmilesToMol(mol_copy, smi.strip()):
                        mol_copy.SetTitle(f"mol_{i}")
                        processed.append(mol_copy)
                        molecule_processed = True
                
                # Update progress when molecule is successfully processed
                if molecule_processed and dots:
                    dots.Update()
                        
            except Exception:
                continue
                
        if dots:
            dots.Total()
                
        return processed

    def _convert_rdkit_to_oe(self, rdkit_mol, mol_idx: int = 0) -> oechem.OEMol:
        """Convert RDKit molecule to OpenEye OEMol"""
        try:
            from rdkit import Chem
            smiles = Chem.MolToSmiles(rdkit_mol)
            oe_mol = oechem.OEMol()
            if oechem.OESmilesToMol(oe_mol, smiles):
                oe_mol.SetTitle(f"mol_{mol_idx}")
                return oe_mol
        except Exception:
            pass
        return None
        
    def _setup_omega(self):
        """Setup OMEGA with proven parameters"""
        opts = oeomega.OEOmegaOptions()
        opts.SetMaxConfs(self.max_conformers)
        opts.SetStrictStereo(False)
        opts.SetFromCT(True)
        opts.SetFixRMS(True)
        opts.SetRMSThreshold(0.5)
        opts.SetEnumRing(True)
        opts.SetRotorOffset(False)
        
        if self.use_gpu and oeomega.OEOmegaIsGPUReady():
            opts.GetTorDriveOptions().SetUseGPU(True)
            opts.SetSampleHydrogens(False)
        else:
            opts.GetTorDriveOptions().SetUseGPU(False)
            opts.SetSampleHydrogens(True)
            
        return oeomega.OEOmega(opts)
        
    def _enumerate_isomers(self, mol) -> List[oechem.OEMol]:
        """Flipper isomer enumeration"""
        opts = oeomega.OEFlipperOptions()
        opts.SetMaxCenters(min(4, self.max_isomers))
        
        isomers = []
        for i, iso in enumerate(oeomega.OEFlipper(mol, opts)):
            if i >= self.max_isomers:
                break
            iso_mol = oechem.OEMol(iso)
            iso_mol.SetTitle(f"{mol.GetTitle()}+{i}")
            isomers.append(iso_mol)
            
        return isomers if isomers else [mol]
        
    def _passes_filters(self, mol) -> bool:
        """Property-based filtering"""
        try:
            heavy_atoms = oechem.OECount(mol, oechem.OEIsHeavy())
            if heavy_atoms > self.max_heavy_atoms or heavy_atoms < 3:
                return False
                
            rot_bonds = oechem.OECount(mol, oechem.OEIsRotor())
            if rot_bonds > self.max_rotatable_bonds:
                return False
                
            return True
        except Exception:
            return False
            
    def _score_multi_query(self, processed_mols) -> dict:
        """Score against multiple queries, return best scores as dict"""
        if not processed_mols:
            return {}
            
        best_scores = {}
        
        for query_file in self.query_files:
            query_scores = self._score_single_query(processed_mols, query_file)
            
            # Take maximum score for each molecule
            for mol_title, score in query_scores.items():
                best_scores[mol_title] = max(score, best_scores.get(mol_title, 0.0))
                
        return best_scores
        
    def _score_single_query(self, processed_mols, query_file: str) -> dict:
        """Score molecules against a single query file and return as dict"""
        if not processed_mols:
            return {}
            
        scores = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write molecules to temporary file
            input_file = self._write_molecules(processed_mols, tmpdir)
            if not input_file:
                return scores
                
            # Execute ROCS
            output_file = self._execute_rocs(query_file, input_file, tmpdir)
            if not output_file:
                return scores
                
            # Parse results and return as dictionary
            scores_array = self._parse_results(output_file, len(processed_mols))
            
            # Convert to dictionary mapping molecule titles to scores
            for i, score in enumerate(scores_array):
                if i < len(processed_mols):
                    mol_title = processed_mols[i].GetTitle()
                    scores[mol_title] = score
                    
        return scores
            
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
            if not shutil.which(self.rocs_binary):
                raise RuntimeError(f"ROCS binary not found: {self.rocs_binary}")
                
            # Execute ROCS with timing
            rocs_timer = oechem.OEWallTimer() if self.show_progress else None
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,
                env=dict(os.environ, OMP_NUM_THREADS="1")
            )
            
            if self.show_progress and rocs_timer and rocs_timer.Elapsed() > 2.0:
                print(f"  ROCS execution: {rocs_timer.Elapsed():.1f}s")
            
            if result.returncode != 0:
                raise RuntimeError(f"ROCS failed with return code {result.returncode}")
                
            if not os.path.exists(output_file):
                raise RuntimeError(f"ROCS output file not created: {output_file}")
                
            if os.path.getsize(output_file) == 0:
                raise RuntimeError(f"ROCS output file is empty: {output_file}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ROCS execution timed out")
        except Exception as e:
            raise RuntimeError(f"ROCS execution failed: {e}")
            
        return output_file
        
    def _build_rocs_command(self, query_file: str, input_file: str, output_file: str) -> List[str]:
        """Build ROCS command with proven parameters"""
        cmd = [
            self.rocs_binary,
            "-query", query_file,
            "-dbase", input_file,
            "-reportfile", output_file,
            "-cutoff", "-1.0",
            "-maxhits", "0",
            "-rankby", "TanimotoCombo",
            "-chemff", "ImplicitMillsDean",
            "-stats", "best",
            "-nostructs",
            "-opt", "true"
        ]
        
        if self.shape_only:
            cmd.extend(["-shapeonly", "true"])
        else:
            cmd.extend(["-optchem", "true"])
            
        return cmd
        
    def _parse_results(self, output_file: str, num_mols: int) -> np.ndarray:
        """Parse ROCS output and return numpy array"""
        scores = np.zeros(num_mols)
        
        if not os.path.exists(output_file):
            return scores
            
        try:
            import pandas as pd
            df = pd.read_csv(output_file, sep='\t')
            
            if df.empty or 'Name' not in df.columns or 'TanimotoCombo' not in df.columns:
                return scores
            
            for _, row in df.iterrows():
                name = row.get('Name', '')
                if name.startswith('mol_'):
                    try:
                        idx = int(name.split('_')[1].split('+')[0])
                        if 0 <= idx < num_mols:
                            tc_score = float(row.get('TanimotoCombo', 0.0))
                            scores[idx] = max(scores[idx], tc_score)
                    except (ValueError, IndexError):
                        continue
                        
        except Exception:
            pass
            
        return scores
        
    def getKey(self) -> str:
        """Return scorer identifier"""
        return "ROCS"

    def _process_large_batch(self, mols) -> np.ndarray:
        """Process large batches by splitting into smaller chunks"""
        num_mols = len(mols)
        result_scores = np.zeros(num_mols)
        
        total_chunks = (num_mols + self.batch_size_limit - 1) // self.batch_size_limit
        
        if self.show_progress:
            print(f"Processing {num_mols} molecules in {total_chunks} chunks...")
        
        # Process in chunks
        for chunk_idx, start_idx in enumerate(range(0, num_mols, self.batch_size_limit)):
            end_idx = min(start_idx + self.batch_size_limit, num_mols)
            chunk = mols[start_idx:end_idx]
            
            if self.show_progress:
                print(f"  Chunk {chunk_idx + 1}/{total_chunks}")
            
            try:
                chunk_scores = self._score_chunk_directly(chunk)
                result_scores[start_idx:end_idx] = chunk_scores
            except Exception:
                continue
                
        return result_scores
    
    def _score_chunk_directly(self, mols) -> np.ndarray:
        """Score a chunk of molecules directly"""
        if not mols:
            return np.zeros(0)
            
        num_input_mols = len(mols)
        result_scores = np.zeros(num_input_mols)
        
        try:
            smiles_list = self._convert_to_smiles(mols)
            processed_mols = self._prepare_molecules_for_cli(smiles_list)
            
            if not processed_mols:
                return result_scores
            
            # Score with retry logic
            scores_dict = self._score_with_retry(processed_mols)
                
            if not scores_dict:
                return result_scores
                
            # Map scores back to chunk indices
            for mol_key, score in scores_dict.items():
                base_mol_key = mol_key.split('+')[0] if '+' in mol_key else mol_key
                
                if base_mol_key.startswith('mol_'):
                    try:
                        chunk_idx = int(base_mol_key.split('_')[1])
                        if 0 <= chunk_idx < num_input_mols:
                            result_scores[chunk_idx] = max(result_scores[chunk_idx], score)
                    except (ValueError, IndexError):
                        continue
                 
        except Exception:
            pass
        
        return result_scores
        
    def _score_with_retry(self, processed_mols) -> dict:
        """Score molecules with retry logic"""
        for attempt in range(1, self.max_retry_attempts + 1):
            try:
                # Multi-query scoring
                if len(self.query_files) == 1:
                    scores_dict = self._score_single_query(processed_mols, self.query_files[0])
                else:
                    scores_dict = self._score_multi_query(processed_mols)
                
                if scores_dict:
                    return scores_dict
                    
            except Exception:
                if attempt == self.max_retry_attempts:
                    break
        
        return {} 