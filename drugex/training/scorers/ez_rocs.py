# (C) 2022 Cadence Design Systems, Inc. (Cadence) 
# All rights reserved.
# TERMS FOR USE OF SAMPLE CODE The software below ("Sample Code") is
# provided to current licensees or subscribers of Cadence products or
# SaaS offerings (each a "Customer").
# Customer is hereby permitted to use, copy, and modify the Sample Code,
# subject to these terms. Cadence claims no rights to Customer's
# modifications. Modification of Sample Code is at Customer's sole and
# exclusive risk. Sample Code may require Customer to have a then
# current license or subscription to the applicable Cadence offering.
# THE SAMPLE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED.  OPENEYE DISCLAIMS ALL WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. In no event shall Cadence be
# liable for any damages or liability in connection with the Sample Code
# or its use.

"""
'Minimal' ROCS scorer implementation using OpenEye tools.
"""

import numpy as np
import pandas as pd
import os
import tempfile
import gc
from pathlib import Path
from openeye import oechem, oeomega, oeff
try:
    from openeye import oeshape, oefastrocs
    FASTROCS_AVAILABLE = True
except ImportError:
    FASTROCS_AVAILABLE = False

from drugex.training.scorers.interfaces import Scorer

# Initialize OpenEye memory pool just once at module import time
_OE_MEMORY_POOL_INITIALIZED = False

def _initialize_oe_memory_pool():
    global _OE_MEMORY_POOL_INITIALIZED
    
    # Check if already initialized in this process via environment variable
    if os.environ.get("OE_MEMORY_POOL_INITIALIZED") == "true":
        _OE_MEMORY_POOL_INITIALIZED = True
        return
        
    if not _OE_MEMORY_POOL_INITIALIZED:
        try:
            oechem.OESetMemPoolMode(oechem.OEMemPoolMode_System)
            _OE_MEMORY_POOL_INITIALIZED = True
            os.environ["OE_MEMORY_POOL_INITIALIZED"] = "true"
            print("OpenEye memory pool initialized in ez_rocs module")
        except Exception as e:
            print(f"Warning: Could not set memory pool mode: {e}")

# Initialize at module import
_initialize_oe_memory_pool()

# ------------------------------------------------------------------------------
#  Adaptive model selection
# ------------------------------------------------------------------------------

class AdaptiveModelSelector:
    """Adaptive model selection for multi-model ROCS scoring."""
    
    def __init__(self, model_paths):
        self.model_paths = model_paths
        self.model_scores = {}
        self.selection_count = 0
        
    def select_models(self, max_models=None):
        """Select best performing models based on historical data."""
        if max_models is None or max_models >= len(self.model_paths):
            return self.model_paths
            
        # Sort by average score (descending)
        sorted_models = sorted(
            self.model_paths,
            key=lambda x: self.model_scores.get(x, 0.0),
            reverse=True
        )
        
        return sorted_models[:max_models]
    
    def update_scores(self, model_path, scores):
        """Update model performance tracking."""
        avg_score = np.mean(scores) if len(scores) > 0 else 0.0
        if model_path in self.model_scores:
            # Rolling average
            self.model_scores[model_path] = (self.model_scores[model_path] + avg_score) / 2
        else:
            self.model_scores[model_path] = avg_score

# ------------------------------------------------------------------------------
#  Helper functions
# ------------------------------------------------------------------------------

def _has_valid_3d_conformers(mol):
    """
    Simple check if an OEMol object has valid 3D conformers.
    
    Parameters
    ----------
    mol : OEMol
        OpenEye molecule object to check
        
    Returns
    -------
    bool
        True if molecule has at least one conformer with 3D coordinates
    """
    if not hasattr(mol, 'NumConfs'):
        return False
        
    if mol.NumConfs() == 0:
        return False
        
    # Check if at least one conformer has 3D coordinates (non-zero Z values)
    for conf in mol.GetConfs():
        coords = oechem.OEFloatArray(mol.GetMaxAtomIdx() * 3)
        conf.GetCoords(coords)
        
        # Check if we have non-zero Z coordinates (indicating 3D structure)
        for i in range(2, len(coords), 3):  # Check every Z coordinate
            if abs(coords[i]) > 1e-6:  # Small threshold for floating point comparison
                return True
                
    return False

def OMEGA(input_file, save_dir, experiment_name, max_confs=200, use_existing_conformers_always=True):
    """
    Generate conformers using OMEGA.
    
    Parameters
    ----------
    input_file : str
        Path to the input file with molecules.
    save_dir : str
        Directory to save the output conformer database.
    experiment_name : str
        Name of the experiment for output file naming.
    max_confs : int, optional
        Maximum number of conformers to generate per molecule (default: 200).
    use_existing_conformers_always : bool, optional
        Whether to use existing conformers when available (default: True).
        
    Returns
    -------
    str
        Path to the generated conformer database.
    """
    output_file = os.path.join(save_dir, f"{experiment_name}_conformers.oeb.gz")
    
    # Set up OMEGA with EXACT alignment to base_rocs.py CLI behavior
    omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts.SetMaxConfs(max_confs)  # Aligned with base_rocs.py: 200 conformers
    omegaOpts.SetStrictStereo(False)
    omegaOpts.SetFromCT(True)
    omegaOpts.SetFixRMS(True)
    omegaOpts.SetRMSThreshold(0.5)
    omegaOpts.SetEnumRing(True)
    omegaOpts.SetRotorOffset(False)
    
    # Force GPU settings to match base_rocs.py behavior
    omegaOpts.GetTorDriveOptions().SetUseGPU(False)  # Force CPU mode for consistency
    omegaOpts.SetSampleHydrogens(True)
    
    omega = oeomega.OEOmega(omegaOpts)
    
    # Process molecules
    ifs = oechem.oemolistream(input_file)
    ofs = oechem.oemolostream(output_file)
    
    mol_count = 0
    conformer_gen_count = 0
    existing_conformer_count = 0
    
    for mol in ifs.GetOEMols():
        # Check if molecule already has valid 3D conformers
        if use_existing_conformers_always and _has_valid_3d_conformers(mol):
            # Use existing conformers, no need to generate new ones
            oechem.OEWriteMolecule(ofs, mol)
            mol_count += 1
            existing_conformer_count += 1
        else:
            # Generate new conformers using OMEGA
            if omega(mol):
                oechem.OEWriteMolecule(ofs, mol)
                mol_count += 1
                conformer_gen_count += 1
    
    ifs.close()
    ofs.close()
    
    if use_existing_conformers_always and existing_conformer_count > 0:
        print(f"OMEGA: Used existing conformers for {existing_conformer_count} molecules, "
              f"generated conformers for {conformer_gen_count} molecules (total: {mol_count})")
    else:
        print(f"OMEGA generated conformers for {mol_count} molecules")
    return output_file


def ROCS(dbname, query_files, save_dir, experiment_name, use_gpu, threads=None):
    """
    Run FastROCS shape comparison on a database of molecules.
    
    Parameters
    ----------
    dbname : str
        Path to the conformer database.
    query_files : list
        List of query file paths.
    save_dir : str
        Directory to save the output CSV file.
    experiment_name : str
        Name of the experiment for output file naming.
    use_gpu : bool
        Whether to use GPU acceleration.
    threads : int, optional
        Number of threads to use for CPU mode.
        
    Returns
    -------
    str
        Path to the output CSV file with results.
    """
    if not FASTROCS_AVAILABLE:
        raise ImportError("FastROCS not available")
    
    outfname = os.path.join(save_dir, f"{experiment_name}_scores.csv")
    
    try:
        # Initialize database with CLI-aligned settings and vROCS compatibility
        # Create color force field first for vROCS approach
        color_ff = oeshape.OEColorForceField()
        color_ff.Init(oeshape.OEColorFFType_ImplicitMillsDeanNoRings)  # Match vROCS reference
        
        # Create database with color force field directly (vROCS approach)
        try:
            db = oefastrocs.OEShapeDatabase(color_ff)
        except:
            # Fallback to standard database creation
            db = oefastrocs.OEShapeDatabase()
        
        opts = oefastrocs.OEShapeDatabaseOptions()
        
        # Force ROCS mode (not FastROCS) to match CLI behavior
        opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_ROCS)
        
        # Configure threading to match CLI behavior
        if use_gpu:
            # GPU mode - single thread like CLI
            db.SetNumOpenThreads(1)
        else:
            # CPU mode - controlled thread count
            if threads is None:
                threads = min(2, max(1, os.cpu_count() - 2)) if os.cpu_count() else 2
            db.SetNumOpenThreads(threads)
            
        # Apply CLI-equivalent database options with vROCS compatibility
        try:
            # Match CLI settings: -cutoff -1.0, -besthits 1, -tanimoto_cutoff 0.0
            opts.SetScoreCutoff(-1.0)  # Return all scores (no cutoff)
            opts.SetLimit(1)  # Return only best hit per molecule (matches -besthits 1)
            # Create proper OEColorForceField object with vROCS-compatible enum
            color_ff = oeshape.OEColorForceField()
            color_ff.Init(oeshape.OEColorFFType_ImplicitMillsDeanNoRings)  # Match vROCS reference
            opts.SetColorForceField(color_ff)  # Match CLI -chemff parameter
            # Enable color optimization for better scoring
            opts.SetColorOptimization(True)
        except AttributeError:
            # Some options may not be available in all FastROCS versions
            pass
        
        # Open the database with the molecule database
        mdb = oechem.OEMolDatabase()
        if not mdb.Open(dbname):
            raise ValueError(f"Failed to open molecule database: {dbname}")
        
        if not db.Open(mdb):
            raise ValueError(f"Failed to open shape database")
    except Exception as e:
        if "license" in str(e).lower():
            print(f"OpenEye license error: {e}")
        else:
            print(f"Error initializing database: {e}")
        raise
    
    # Create a single output file for all queries
    with open(outfname, 'w') as f:
        # Write header
        f.write("TITLE,QueryFile,TanimotoCombo,ShapeTanimoto,ColorTanimoto\n")
    
    # Process each query file
    for qfname in query_files:
        # Ensure qfname is a string, not a list
        if isinstance(qfname, (list, tuple)):
            current_file = qfname[0]
            print(f"Warning: Expected string path, got list. Using first: {current_file}")
        else:
            current_file = qfname
            
        # Read query
        query = oeshape.OEShapeQuery()
        if not oeshape.OEReadShapeQuery(current_file, query):
            print(f"Warning: Could not read shape query: {current_file}")
            continue
            
        query_name = os.path.basename(current_file)
        
        try:
            # Score all molecules
            score_count = 0
            with open(outfname, 'a') as f:
                print(f"Scoring with query: {query_name}")
                for score in db.GetSortedScores(query, opts):
                    mol_idx = score.GetMolIdx()
                    tanimoto_combo = score.GetTanimotoCombo()
                    shape_tanimoto = score.GetShapeTanimoto()
                    color_tanimoto = score.GetColorTanimoto()
                    
                    # Get molecule title - correct API usage
                    mol = oechem.OEGraphMol()
                    mdb.GetMolecule(mol, mol_idx)
                    title = mol.GetTitle()
                    
                    # Write results
                    f.write(f"{title},{query_name},{tanimoto_combo},{shape_tanimoto},{color_tanimoto}\n")
                    score_count += 1
                    
                print(f"Wrote {score_count} scores for query {query_name}")
        except Exception as e:
            if "license" in str(e).lower():
                print(f"OpenEye license error in ROCS scoring: {e}")
                raise
            else:
                print(f"Error in ROCS scoring: {e}")
            
    # Clean up to reduce memory usage
    # mdb.Close()  # OEMolDatabase doesn't have a Close() method in this version
    # db.Close()   # OEShapeDatabase doesn't have a Close() method in this version
    # Release references to allow garbage collection instead
    db = None
    mdb = None
    gc.collect()
    
    return outfname


# ------------------------------------------------------------------------------
#  Main scorer class
# ------------------------------------------------------------------------------

class RocsScorer(Scorer):
    """
    Simplified ROCS scorer implementation for basic usage and educational purposes.
    
    This implementation prioritizes clarity and ease of use over performance.
    Suitable for small to medium molecule sets and development environments.
    """
    
    def __init__(self, sq_model_path=None, query_file=None, experiment_name="rocs_experiment", 
                 score_type="TanimotoCombo", use_gpu=False, max_isomers=4, max_rot_bonds=15, 
                 max_heavy_atoms=35, max_conformers=200, cpu_processes=None,
                 top_n_models=None, parallel_execution=True, use_existing_conformers_always=True, save_conformers=None):
        """
        Initialize the RocsScorer.
        
        Parameters
        ----------
        sq_model_path : str or list, optional
            Path to ROCS query file(s) (.sq format)
        query_file : str or list, optional  
            Alternative parameter name for sq_model_path
        experiment_name : str, optional
            Name for temporary files and experiment tracking
        score_type : str, optional
            Type of score to extract ("TanimotoCombo", "ShapeTanimoto", "ColorTanimoto")
        use_gpu : bool, optional
            Whether to use GPU acceleration if available
        max_isomers : int, optional
            Maximum number of isomers to enumerate per molecule (aligned with base_rocs.py)
        max_rot_bonds : int, optional
            Maximum rotatable bonds threshold (aligned with base_rocs.py)
        max_heavy_atoms : int, optional
            Maximum heavy atoms threshold (aligned with base_rocs.py)
        max_conformers : int, optional
            Maximum conformers per molecule (aligned with base_rocs.py CLI default: 200)
        cpu_processes : int, optional
            Number of CPU processes to use
        top_n_models : int, optional
            Use only top N performing models (for multi-model setups)
        parallel_execution : bool, optional
            Whether to enable parallel processing
        use_existing_conformers_always : bool, optional
            Whether to use existing 3D conformers when available instead of generating new ones (default: True)
        save_conformers : str, optional
            Directory to save generated conformers (default: None, no saving).
        """
        # Handle both sq_model_path and query_file parameters for compatibility
        if sq_model_path is not None:
            if isinstance(sq_model_path, (list, tuple)):
                self.qfnames = list(sq_model_path)
            else:
                self.qfnames = [sq_model_path]
        elif query_file is not None:
            if isinstance(query_file, (list, tuple)):
                self.qfnames = list(query_file)
            else:
                self.qfnames = [query_file]
        else:
            raise ValueError("Either sq_model_path or query_file must be provided")
            
        # Store a single query file for backward compatibility
        self.query_file = self.qfnames[0]
        
        # Initialize other parameters
        self.experiment_name = experiment_name
        self.score_type = score_type
        self.use_gpu = use_gpu and FASTROCS_AVAILABLE
        self.max_isomers = max_isomers
        self.max_rot_bonds = max_rot_bonds
        self.max_heavy_atoms = max_heavy_atoms
        self.max_conformers = max_conformers
        self.top_n_models = top_n_models
        self.parallel_execution = parallel_execution
        self.use_existing_conformers_always = use_existing_conformers_always
        self.save_conformers = save_conformers
        
        # Set up adaptive model selection if needed
        self.model_selector = None
        if top_n_models is not None and len(self.qfnames) > 1:
            self.model_selector = AdaptiveModelSelector(self.qfnames)
        
        # Automatically determine CPU processes if not specified
        if cpu_processes is None:
            # Conservative default to avoid resource exhaustion
            self.cpu_processes = min(2, max(1, os.cpu_count() - 2)) if os.cpu_count() else 2
        else:
            self.cpu_processes = cpu_processes
            
        # Set up cache directory
        self._cache_dir = os.path.join(tempfile.gettempdir(), "rocs_scorer_cache")
        os.makedirs(self._cache_dir, exist_ok=True)
        
        # Validate all query files individually
        valid_models = []
        for qfname in self.qfnames:
            if not os.path.exists(qfname):
                print(f"Warning: Query file not found: {qfname}")
                continue
                
            # Validate the query file one at a time
            try:
                query = oeshape.OEShapeQuery()
                # Ensure qfname is a string, not a list
                if isinstance(qfname, (list, tuple)):
                    current_file = qfname[0]
                    print(f"Warning: Expected string path, got list. Using first: {current_file}")
                else:
                    current_file = qfname
                    
                if not oeshape.OEReadShapeQuery(current_file, query):
                    print(f"Warning: Invalid shape query file: {current_file}")
                    continue
                    
                valid_models.append(current_file)
            except Exception as e:
                if "license" in str(e).lower():
                    print(f"OpenEye license error: {e}")
                    raise
                else:
                    print(f"Error validating query file {qfname}: {e}")
                    continue
                
        if not valid_models:
            raise ValueError("No valid query files found. Please provide at least one valid .sq file.")
            
        # Update qfnames to contain only valid models
        self.qfnames = valid_models
        self.query_file = self.qfnames[0]  # Update primary query file
                
        # Set up OpenEye
        os.environ["OE_SILENT"] = "true"
        
        # Log configuration
        gpu_str = "GPU" if self.use_gpu else f"CPU ({self.cpu_processes} processes)"
        model_str = f"{len(self.qfnames)} shape query files"
        print(f"ROCS scorer initialized using {gpu_str} mode with {model_str}")

    def getScores(self, mols):
        """
        Score molecules using ROCS.
        
        Parameters
        ----------
        mols : list
            List of molecules (SMILES strings or OEMol objects)
            
        Returns
        -------
        np.ndarray
            Array of ROCS scores
        """
        if not mols:
            return np.zeros(0)
            
        # Ensure consistent molecule naming for input
        if any(hasattr(mol, 'GetTitle') for mol in mols):
            mols = self._ensure_consistent_molecule_naming(mols)
            
        # Keep original molecules for conformer detection
        # Convert SMILES to a consistent format but preserve OEMol objects with conformers
        processed_mols = []
        for mol in mols:
            if isinstance(mol, str):
                processed_mols.append(mol)
            elif hasattr(mol, 'GetTitle'):  # OEMol object
                # Check if this OEMol has conformers and we want to use them
                if self.use_existing_conformers_always and _has_valid_3d_conformers(mol):
                    processed_mols.append(mol)  # Keep the OEMol object
                else:
                    # Convert to SMILES for standard processing
                    smi = oechem.OECreateSmiString(mol)
                    processed_mols.append(smi)
            else:
                print(f"Warning: Unsupported molecule type: {type(mol)}")
                processed_mols.append("")
        
        # Use multiple models if available
        if len(self.qfnames) > 1:
            all_scores = []
            active_models = self.qfnames
            
            # Use adaptive model selection if configured
            if self.model_selector and self.top_n_models:
                active_models = self.model_selector.select_models(self.top_n_models)
            
            for model_path in active_models:
                model_scores = self._score_with_single_model(processed_mols, model_path)
                all_scores.append(model_scores)
                
                # Update model selector if available
                if self.model_selector:
                    self.model_selector.update_scores(model_path, model_scores)
            
            # Take maximum score across all models for each molecule
            if all_scores:
                combined_scores = np.maximum.reduce(all_scores)
                return combined_scores
            else:
                return np.zeros(len(processed_mols))
        else:
            # Single model scoring
            return self._score_with_single_model(processed_mols, self.qfnames[0])

    def _score_with_single_model(self, mols, model_path):
        """Score molecules with a single ROCS model."""
        if not mols:
            return np.zeros(0)
            
        # Create directory for this scoring run
        if self.save_conformers is None:
            # Create temporary directory for all files
            conf_dir = tempfile.mkdtemp(prefix="ez_rocs_")
        else:
            conf_dir = self.save_conformers
            os.makedirs(conf_dir, exist_ok=True)
        
        try:
            # Separate molecules with conformers from those without
            molecules_with_conformers = []
            molecules_without_conformers = []
            mol_indices = []  # Track original indices
            
            for i, mol in enumerate(mols):
                if isinstance(mol, str):
                    # SMILES string
                    molecules_without_conformers.append((i, mol))
                elif hasattr(mol, 'GetTitle') and _has_valid_3d_conformers(mol):
                    # OEMol with conformers
                    molecules_with_conformers.append((i, mol))
                elif hasattr(mol, 'GetTitle'):
                    # OEMol without conformers - convert to SMILES
                    smi = oechem.OECreateSmiString(mol)
                    molecules_without_conformers.append((i, smi))
                else:
                    print(f"Warning: Unsupported molecule type at index {i}: {type(mol)}")
                    molecules_without_conformers.append((i, ""))
            
            # Create molecular database file
            mol_db_file = os.path.join(conf_dir, "molecules.oeb.gz")
            ofs = oechem.oemolostream(mol_db_file)
            
            mol_index_map = {}  # Map from molecule DB index to original index
            current_db_index = 0
            
            # Process molecules with existing conformers
            if molecules_with_conformers and self.use_existing_conformers_always:
                print(f"Processing {len(molecules_with_conformers)} molecules with existing conformers")
                for orig_idx, mol in molecules_with_conformers:
                    # Apply molecular filters
                    try:
                        heavy_count = oechem.OECount(mol, oechem.OEIsHeavy())
                        if heavy_count > self.max_heavy_atoms:
                            continue
                            
                        rot_bonds = oechem.OECount(mol, oechem.OEIsRotor())
                        if rot_bonds > self.max_rot_bonds:
                            continue
                        
                        # Set title for tracking
                        mol.SetTitle(f"conf_{current_db_index}")
                        mol_index_map[current_db_index] = orig_idx
                        
                        # Write molecule with existing conformers
                        oechem.OEWriteMolecule(ofs, mol)
                        current_db_index += 1
                        
                    except Exception as e:
                        print(f"Error processing molecule with conformers at index {orig_idx}: {e}")
                        continue
            
            # Process molecules without conformers (SMILES or OEMol without conformers)
            isomer_count = 0
            if molecules_without_conformers:
                # Create isomers file for OMEGA processing
                isomers_file = os.path.join(conf_dir, "isomers.smi")
                
                # Apply molecular filters and generate isomers
                valid_molecules = []
                with open(isomers_file, 'w') as f:
                    for orig_idx, smi in molecules_without_conformers:
                        if not smi or not smi.strip():
                            continue
                            
                        try:
                            mol = oechem.OEGraphMol()
                            if not oechem.OESmilesToMol(mol, smi.strip()):
                                continue
                                
                            # Apply filtering thresholds
                            heavy_count = oechem.OECount(mol, oechem.OEIsHeavy())
                            if heavy_count > self.max_heavy_atoms:
                                continue
                                
                            rot_bonds = oechem.OECount(mol, oechem.OEIsRotor())
                            if rot_bonds > self.max_rot_bonds:
                                continue
                                
                            valid_molecules.append((orig_idx, smi.strip()))
                            
                        except Exception as e:
                            print(f"Error processing molecule {smi}: {e}")
                            continue
                
                # Generate isomers and write to file
                with open(isomers_file, 'w') as f:
                    for mi, (orig_idx, smi) in enumerate(valid_molecules):
                        try:
                            mol = oechem.OEGraphMol()
                            if not oechem.OESmilesToMol(mol, smi):
                                continue
                                
                            # Generate isomers
                            flipper_opts = oeomega.OEFlipperOptions()
                            mol_isomer_count = 0
                            
                            for iso in oeomega.OEFlipper(mol, flipper_opts):
                                if mol_isomer_count >= self.max_isomers:
                                    break
                                iso_smi = oechem.OEMolToSmiles(iso)
                                # Use a unique identifier that includes the current_db_index
                                msid = f"smi_{current_db_index + mi}+{mol_isomer_count}"
                                f.write(f"{iso_smi}\t{msid}\n")
                                isomer_count += 1
                                mol_isomer_count += 1
                                
                                # Map the database index to original index (for first isomer)
                                if mol_isomer_count == 1:
                                    mol_index_map[current_db_index + mi] = orig_idx
                                    
                        except Exception as e:
                            print(f"Error generating isomer for {smi}: {e}")
                
                if isomer_count > 0:
                    # Generate conformers using OMEGA and append to the molecular database
                    conformer_file = OMEGA(isomers_file, conf_dir, self.experiment_name, self.max_conformers, False)  # Don't skip conformers for these
                    
                    # Append the conformers to the existing molecular database
                    ifs_conf = oechem.oemolistream(conformer_file)
                    for mol in ifs_conf.GetOEMols():
                        oechem.OEWriteMolecule(ofs, mol)
                    ifs_conf.close()
                    
                    # Clean up conformer file
                    try:
                        os.remove(conformer_file)
                    except:
                        pass
            
            ofs.close()
            
            # Check if we have any molecules to score
            if current_db_index == 0 and (not molecules_without_conformers or isomer_count == 0):
                return np.zeros(len(mols))

            # Run ROCS with the combined molecular database
            threads = None if self.use_gpu else self.cpu_processes
            rocs_output = ROCS(mol_db_file, [model_path], conf_dir, self.experiment_name, self.use_gpu, threads)
            
            # Parse results
            try:
                df = pd.read_csv(rocs_output)
                scores = np.zeros(len(mols))
                
                if not df.empty:
                    # Extract scores by molecule index - simplified since FastROCS returns only best hit per molecule
                    for _, row in df.iterrows():
                        title = str(row['TITLE'])
                        
                        try:
                            if title.startswith('conf_'):
                                # Molecule with existing conformers
                                db_idx = int(title.split('_')[1])
                            elif title.startswith('smi_') and '+' in title:
                                # Molecule processed through OMEGA
                                db_idx_str = title.split('+')[0].replace('smi_', '')
                                db_idx = int(db_idx_str)
                            else:
                                continue
                                
                            if db_idx in mol_index_map:
                                orig_idx = mol_index_map[db_idx]
                                score_value = float(row[self.score_type])
                                scores[orig_idx] = score_value  # Direct assignment since we get only best score
                                
                        except (ValueError, IndexError, KeyError) as e:
                            continue
                
                return scores
                
            except Exception as e:
                print(f"Error processing ROCS results: {e}")
                return np.zeros(len(mols))
            
        except Exception as e:
            print(f"Error in ROCS scoring: {e}")
            return np.zeros(len(mols))
            
        finally:
            if self.save_conformers is None:
                # Clean up all temporary files
                for file_path in [os.path.join(conf_dir, f) for f in os.listdir(conf_dir)]:
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"Error removing file {file_path}: {e}")
                
                try:
                    os.rmdir(conf_dir)
                except Exception as e:
                    print(f"Error removing directory {conf_dir}: {e}")
                
            # Force cleanup
            gc.collect()

    def getKey(self):
        """Return scorer identifier."""
        return "ROCS"
    
    def _extract_base_molecule_name(self, full_name: str) -> str:
        """
        Extract base molecule name from conformer-specific names.
        This should match the implementation in BaseROCSScorer for consistency.
        """
        if not full_name:
            return full_name
            
        # Remove common conformer suffixes
        base_name = full_name
        
        # Pattern 1: name_conf_number
        if '_conf_' in base_name:
            base_name = base_name.split('_conf_')[0]
        
        # Pattern 2: name+isomer_conf_number  
        if '+' in base_name and '_conf_' in full_name:
            base_name = base_name.split('+')[0]
            
        # Pattern 3: Remove trailing _number if it looks like a conformer ID
        import re
        if re.match(r'.*_\d+$', base_name) and not base_name.startswith('conf_'):
            # Only remove if the number part is likely a conformer ID (not part of the name)
            parts = base_name.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_name = parts[0]
        
        return base_name

    def _ensure_consistent_molecule_naming(self, mols):
        """
        Ensure consistent molecule naming for input molecules.
        This method sets consistent base names for molecules before processing.
        """
        for i, mol in enumerate(mols):
            if hasattr(mol, 'GetTitle') and hasattr(mol, 'SetTitle'):
                current_title = mol.GetTitle()
                if current_title:
                    base_name = self._extract_base_molecule_name(current_title)
                    mol.SetTitle(base_name)
                else:
                    mol.SetTitle(f"mol_{i}")
        return mols
