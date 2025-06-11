#!/usr/bin/env python3
import os
import shutil
import subprocess
import tempfile
import numpy as np
from typing import Union, List

try:
    from openeye import oechem, oeomega
    OE_AVAILABLE = True
except ImportError:
    OE_AVAILABLE = False

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from drugex.training.scorers.interfaces import Scorer

class ROCSCLIScorer(Scorer):
    """Minimalist ROCS CLI scorer that produces a tsv with results"""
    
    def __init__(
        self,
        query_files: Union[str, List[str]],
        score_type: str = "TanimotoCombo",
        max_conformers: int = 200,
        shape_only: bool = False,
        optimize: bool = True,
        color_optimize: bool = True,
        color_force_field: str = "ImplicitMillsDean",
        binary_path: str = None,
        output_file: str = None
    ):
        super().__init__()
        if not OE_AVAILABLE:
            raise ImportError("OpenEye oechem toolkit is required")
        
        # Convert query files to absolute paths
        if isinstance(query_files, str):
            self.query_files = [os.path.abspath(query_files)]
        else:
            self.query_files = [os.path.abspath(qf) for qf in query_files]

        self.score_type = score_type
        self.max_conformers = max_conformers
        self.shape_only = shape_only
        self.optimize = optimize
        self.color_optimize = color_optimize
        self.color_force_field = color_force_field
        self.binary_path = binary_path or "rocs"
        self.output_file = output_file  # Custom output file path
        
        if not shutil.which(self.binary_path):
            raise FileNotFoundError(f"ROCS binary not found: {self.binary_path}")

    def __call__(self, mols, frags=None) -> np.ndarray:
        """DrugEx interface - score molecules and return numpy array"""
        if not isinstance(mols, list):
            mols = [mols]
            
        # Convert various molecule types to SMILES
        smiles_list = []
        for mol in mols:
            if mol is None:
                smiles_list.append("")
            elif isinstance(mol, str):
                smiles_list.append(mol)
            elif hasattr(mol, 'GetTitle'):  # OpenEye molecule
                smi = oechem.OECreateSmiString(mol)
                smiles_list.append(smi if smi else "")
            elif RDKIT_AVAILABLE and hasattr(mol, 'GetNumAtoms'):  # RDKit molecule
                try:
                    smi = Chem.MolToSmiles(mol)
                    smiles_list.append(smi if smi else "")
                except:
                    smiles_list.append("")
            else:
                smiles_list.append("")
        
        return self._score_smiles_list(smiles_list)

    def _score_smiles_list(self, smiles_list: List[str]) -> np.ndarray:
        """Score a list of SMILES strings and return numpy array"""
        num_mols = len(smiles_list)
        scores = np.zeros(num_mols)
        
        if not smiles_list:
            return scores
            
        # Filter valid SMILES and track indices
        valid_molecules = []
        valid_indices = []
        
        for i, smi in enumerate(smiles_list):
            if smi and smi.strip():
                try:
                    # Quick validation
                    mol = oechem.OEMol()
                    if oechem.OESmilesToMol(mol, smi):
                        valid_molecules.append((smi, i))
                        valid_indices.append(i)
                except:
                    continue
        
        if not valid_molecules:
            return scores
            
        # Create temporary files for processing
        with tempfile.TemporaryDirectory() as tmpdir:
            # Prepare molecules with conformers
            processed_mols = self._prepare_molecules_for_cli(valid_molecules, tmpdir)
            if not processed_mols:
                return scores
                
            # Write molecules to OEB file
            input_file = os.path.join(tmpdir, "input.oeb")
            if not self._write_molecules_to_file(processed_mols, input_file):
                return scores
                
            # Run ROCS CLI
            output_file = os.path.join(tmpdir, "output.tsv")
            if self.run_rocs(input_file, output_file):
                # Parse results and map back to original indices
                mol_scores = self._parse_rocs_output(output_file)
                
                # Map scores back to original positions
                for smi, orig_idx in valid_molecules:
                    if orig_idx in mol_scores:
                        scores[orig_idx] = mol_scores[orig_idx]
                        
        return scores

    def _prepare_molecules_for_cli(self, valid_molecules: List[tuple], tmpdir: str) -> List[oechem.OEMol]:
        """Prepare molecules with conformers for CLI scoring"""
        processed = []
        
        # Set up OMEGA for conformer generation
        omega = oeomega.OEOmega()
        omega.SetMaxConfs(self.max_conformers)
        
        for smi, orig_idx in valid_molecules:
            try:
                mol = oechem.OEMol()
                if not oechem.OESmilesToMol(mol, smi):
                    continue
                    
                # Set title to track original index
                mol.SetTitle(f"mol_{orig_idx}")
                
                # Generate conformers safely
                try:
                    if omega(mol):
                        processed.append(mol)
                except Exception as e:
                    print(f"Warning: OMEGA failed for molecule {orig_idx}: {e}")
                    # Try without conformers as fallback
                    try:
                        mol_copy = oechem.OEMol(mol)
                        mol_copy.SetTitle(f"mol_{orig_idx}")
                        processed.append(mol_copy)
                    except:
                        continue
                        
            except Exception as e:
                print(f"Warning: Failed to prepare molecule {orig_idx}: {e}")
                continue
                
        return processed

    def _write_molecules_to_file(self, molecules: List[oechem.OEMol], filename: str) -> bool:
        """Write molecules to OEB file"""
        try:
            with oechem.oemolostream() as ofs:
                if not ofs.open(filename):
                    return False
                    
                for mol in molecules:
                    oechem.OEWriteMolecule(ofs, mol)
                    
            return True
        except Exception as e:
            print(f"Error writing molecules: {e}")
            return False

    def _parse_rocs_output(self, output_file: str) -> dict:
        """Parse ROCS output and return scores mapped by original index"""
        scores = {}
        
        try:
            import pandas as pd
            df = pd.read_csv(output_file, sep='\t')
            
            if 'Name' in df.columns and self.score_type in df.columns:
                for _, row in df.iterrows():
                    name = row['Name']
                    score = float(row[self.score_type])
                    
                    # Extract original index from molecule name
                    if name.startswith('mol_'):
                        try:
                            orig_idx = int(name.split('_')[1])
                            scores[orig_idx] = max(score, scores.get(orig_idx, 0.0))
                        except:
                            continue
                            
        except Exception as e:
            print(f"Warning: Failed to parse ROCS output: {e}")
            
        return scores
    
    def run_rocs(self, input_file, output_tsv, temp_file=None, file_label=0):
        """Run ROCS CLI command to score molecules"""
        # Use a temporary file if provided, otherwise use the final output
        target_output = temp_file or output_tsv
        # Extract the directory from output_tsv
        output_dir = os.path.dirname(target_output) or "."
        
        # DEBUG: Print file paths being used
        print(f"DEBUG CLI: Input file: {input_file}")
        print(f"DEBUG CLI: Query file: {self.query_files[0]}")
        print(f"DEBUG CLI: Output file: {target_output}")
        print(f"DEBUG CLI: File label: {file_label}")
        
        # Handle compressed files by decompressing them first
        actual_input = input_file
        temp_input = None
        
        if input_file.endswith('.gz'):
            import gzip
            import tempfile
            # Create temporary uncompressed file
            with tempfile.NamedTemporaryFile(suffix='.oeb', delete=False) as tmp:
                temp_input = tmp.name
            
            print(f"Decompressing {os.path.basename(input_file)} for CLI processing...")
            with gzip.open(input_file, 'rb') as f_in:
                with open(temp_input, 'wb') as f_out:
                    f_out.write(f_in.read())
            actual_input = temp_input
            print(f"DEBUG CLI: Decompressed to: {actual_input}")
        
        cmd = [
            self.binary_path, 
            "-query", self.query_files[0], 
            "-dbase", actual_input,
            "-report", "one", 
            "-reportfile", target_output,
            "-prefix", "rocs", 
            "-outputdir", output_dir,  # Add output directory
            "-maxconfs", str(self.max_conformers),
            "-rankby", self.score_type, 
            "-chemff", self.color_force_field,
            "-cutoff", "-1.0",  # Return all molecules (no cutoff)
            "-maxhits", "0",    # Return all molecules (overrides besthits)
            "-tanimoto_cutoff", "0.0",
            "-stats", "best", 
            "-nostructs"
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
            
        # Remove conditional adds since we're setting all parameters explicitly
        
        print(f"DEBUG CLI: Executing ROCS command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stderr:
            print(f"ROCS stderr: {result.stderr}")
        
        # DEBUG: Print first few lines of stdout
        if result.stdout:
            stdout_lines = result.stdout.split('\n')[:10]
            print(f"DEBUG CLI: ROCS stdout (first 10 lines):")
            for line in stdout_lines:
                print(f"  {line}")
        
        # Clean up temporary input file if created
        if temp_input and os.path.exists(temp_input):
            os.remove(temp_input)
        
        if result.returncode != 0:
            print(f"ROCS execution failed with code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
        
        # Verify output file was created and has content
        if not os.path.exists(target_output):
            print(f"ERROR: ROCS did not create output file {target_output}")
            return False
        
        file_size = os.path.getsize(target_output)
        print(f"ROCS output file size: {file_size} bytes")
        
        # DEBUG: Print first few lines of output file
        try:
            with open(target_output, 'r') as f:
                output_lines = f.readlines()[:5]
            print(f"DEBUG CLI: Output file first 5 lines:")
            for i, line in enumerate(output_lines):
                print(f"  {i+1}: {line.strip()}")
        except Exception as e:
            print(f"DEBUG CLI: Could not read output file: {e}")
        
        # Post-process the output to fix ranking
        self._fix_ranking(target_output, file_label)
        
        return True
    
    def _fix_ranking(self, output_file, file_label=0):
        """Fix the ranking in the ROCS output file"""
        try:
            import pandas as pd
            df = pd.read_csv(output_file, sep='\t')
            
            # Remove any leading/trailing whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Sort by TanimotoCombo in descending order and assign proper ranks
            if 'TanimotoCombo' in df.columns:
                df = df.sort_values(by='TanimotoCombo', ascending=False).reset_index(drop=True)
                df['Rank'] = df.index + 1
                
                # Add Active column if not present
                if 'Active' not in df.columns:
                    df['Active'] = file_label
                
                # Save back to file
                df.to_csv(output_file, sep='\t', index=False)
                print(f"Fixed ranking for {len(df)} molecules")
            else:
                print("Warning: TanimotoCombo column not found, skipping ranking fix")
                
        except Exception as e:
            print(f"Warning: Failed to fix ranking: {e}")
    
    def getKey(self):
        """Required implementation of Scorer abstract method"""
        return "ROCS"