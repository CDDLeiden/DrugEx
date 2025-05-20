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

Key features:
- Simplified implementation suitable for basic usage and learning
- Easy to understand with minimal dependencies
- Compatible with both CPU and GPU modes (via FastROCS)
- Works well for small to medium molecule sets
- Lower memory footprint than more complex implementations
- Ideal for initial benchmarking and testing

This implementation prioritizes simplicity over performance and is
recommended for development environments, educational purposes,
or when processing modest numbers of molecules.
"""

import numpy as np
import pandas as pd
import os
import tempfile
import gc
from pathlib import Path
from openeye import oechem, oeomega
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
    """
    Class to adaptively select top-performing shape models during training.
    Tracks model performance and provides a mechanism to focus on the best models.
    
    https://openreview.net/forum?id=2M9CUnYnBA
    
    https://www.nature.com/articles/s41598-019-41594-3
    
    Uses Exponential Moving Average (EMA) to balance between recent performance 
    and historical data, allowing the system to adapt to changing molecular distributions
    while maintaining stability in model selection.
    
    TODO:
    - Tune alpha (EMA smoothing) and top_n_models for your dataset via validation.
    - Add a warm-up period (e.g., use all models for first 5–10 batches).
    - Periodically review model selection stats for stability and diversity.
    """
    
    def __init__(self, model_paths):
        """
        Initialize the adaptive model selector with a list of model paths.
        
        Parameters
        ----------
        model_paths : List[str]
            List of paths to shape query (.sq) files
        """
        self.model_paths = model_paths
        self.model_scores = {model: 0.0 for model in model_paths}
        self.usage_counts = {model: 0 for model in model_paths}
        self.history = []  # Track score history for trending
    
    def update_model_scores(self, new_scores):
        """
        Update model scores with new data using exponential moving average (EMA).
        
        EMA gives more weight to recent scores while maintaining influence from historical
        performance. This creates a balance between stability and adaptability in model selection,
        helping to identify consistently high-performing models over time while remaining
        responsive to recent improvements.
        
        Formula: EMA = α * current_score + (1-α) * previous_EMA
        
        Parameters
        ----------
        new_scores : Dict[str, float]
            Dictionary mapping model paths to their average scores
        """
        alpha = 0.3  # Smoothing factor - higher means more weight on recent scores
        
        for model, score in new_scores.items():
            if model in self.model_scores:
                # Update with exponential moving average
                # EMA calculation gives 30% weight to new scores and 70% to historical performance
                # This balances responsiveness to new data with stability in model selection
                old_score = self.model_scores[model]
                self.model_scores[model] = alpha * score + (1 - alpha) * old_score
                # Increment usage counter
                self.usage_counts[model] += 1
        
        # Store history for potential analysis
        self.history.append(self.model_scores.copy())
        
        # Keep history size manageable
        if len(self.history) > 20:
            self.history.pop(0)
    
    def get_active_models(self, top_n=None):
        """
        Get the top N performing models based on current scores.
        
        Parameters
        ----------
        top_n : int, optional
            Number of top models to return. If None, returns all models.
            
        Returns
        -------
        List[str]
            List of model paths for the top performing models
        """
        if top_n is None or top_n >= len(self.model_paths):
            return self.model_paths
            
        # Sort models by score (descending)
        sorted_models = sorted(
            self.model_paths,
            key=lambda model: self.model_scores.get(model, 0.0),
            reverse=True
        )
        
        return sorted_models[:top_n]
    
    def get_model_stats(self):
        """
        Get statistics about model performance.
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary with model paths as keys and performance stats as values
        """
        stats = {}
        for model in self.model_paths:
            stats[model] = {
                'score': self.model_scores.get(model, 0.0),
                'usage': self.usage_counts.get(model, 0),
                'relative_score': self.model_scores.get(model, 0.0) / max(max(self.model_scores.values()), 0.001)
            }
        return stats

def OMEGA(input_file, experiment_name, max_confs=10):
    """
    Generate conformers using OMEGA.
    
    Parameters
    ----------
    input_file : str
        Path to the input file with molecules.
    experiment_name : str
        Name of the experiment for output file naming.
    max_confs : int, optional
        Maximum number of conformers to generate per molecule (default: 10).
        
    Returns
    -------
    str
        Path to the generated conformer database.
    """
    # Use temporary directory for output files
    temp_dir = tempfile.gettempdir()
    dbname = os.path.join(temp_dir, f"{experiment_name}_conformers.oeb.gz")
    
    # Set up OMEGA options
    omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts.SetMaxConfs(max_confs)  # Use configurable value
    omegaOpts.SetStrictStereo(False)  # Don't enforce stereo constraints
    
    # Create omega
    omega = oeomega.OEOmega(omegaOpts)
    
    # Read molecules from input file
    ifs = oechem.oemolistream()
    if not ifs.open(input_file):
        raise ValueError(f"Cannot open input file: {input_file}")
        
    # Create output file
    ofs = oechem.oemolostream()
    if not ofs.open(dbname):
        raise ValueError(f"Cannot create output file: {dbname}")
    
    # Generate conformers - fixed to correctly handle OEMol objects for omega
    for mol_iter in ifs.GetOEMols():
        # Create a copy of the molecule for conformer generation
        mol = oechem.OEMol(mol_iter)
        try:
            if omega(mol):  # Correct calling pattern for omega
                oechem.OEWriteMolecule(ofs, mol)
        except oechem.OELicenseError as e:
            print(f"OpenEye license error in OMEGA: {e}")
            raise
        except Exception as e:
            print(f"Error generating conformers: {e}")
    
    ifs.close()
    ofs.close()
    
    return dbname

def ROCS(dbname, query_files, experiment_name, use_gpu, threads=None):
    """
    Run FastROCS shape comparison on a database of molecules.
    
    Parameters
    ----------
    dbname : str
        Path to the conformer database.
    query_files : list
        List of query file paths.
    experiment_name : str
        Name of the experiment for output file naming.
    use_gpu : bool
        Whether to use GPU acceleration if available.
    threads : int, optional
        Number of CPU threads to use when use_gpu is False.
        
    Returns
    -------
    str
        Path to the output CSV file with results.
    """
    # Use temporary directory for output files
    temp_dir = tempfile.gettempdir()
    outfname = os.path.join(temp_dir, f"{experiment_name}_results.csv")
    
    try:
        # Create molecule database
        mdb = oechem.OEMolDatabase()
        if not mdb.Open(dbname):
            raise ValueError(f"Cannot open database: {dbname}")
        
        # Set up shape database
        db = oefastrocs.OEShapeDatabase()
        if use_gpu:
            # GPU mode settings
            db.SetNumOpenThreads(1)  # Single thread for GPU mode
            opts = oefastrocs.OEShapeDatabaseOptions()
            opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_FastROCS)
        else:
            # CPU mode settings with controlled thread count
            if threads is None:
                # Default to 2 threads if not specified, leaving some cores for system
                threads = min(2, max(1, os.cpu_count() - 2)) if os.cpu_count() else 2
            db.SetNumOpenThreads(threads)
            opts = oefastrocs.OEShapeDatabaseOptions()
            opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_ROCS)  # Explicitly set ROCS mode for CPU
        
        # Open the database with the molecule database
        if not db.Open(mdb):
            raise ValueError(f"Failed to open shape database")
    except oechem.OELicenseError as e:
        print(f"OpenEye license error: {e}")
        raise
    except Exception as e:
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
            print(f"Warning: Cannot read query file: {current_file}")
            continue
        
        try:
            # Append to existing file instead of overwriting
            with open(outfname, 'a') as f:
                # Process each score
                for score in db.GetSortedScores(query, opts):
                    mol_idx = score.GetMolIdx()
                    mol = oechem.OEGraphMol()
                    
                    if mdb.GetMolecule(mol, mol_idx):
                        title = mol.GetTitle()
                        tanimoto_combo = score.GetTanimotoCombo()
                        shape_tanimoto = score.GetShapeTanimoto()
                        color_tanimoto = score.GetColorTanimoto()
                        
                        # Write score to file with query file name
                        query_name = os.path.basename(current_file)
                        f.write(f"{title},{query_name},{tanimoto_combo},{shape_tanimoto},{color_tanimoto}\n")
        except oechem.OELicenseError as e:
            print(f"OpenEye license error in ROCS scoring: {e}")
            raise
        except Exception as e:
            print(f"Error in ROCS scoring: {e}")
            
    # Clean up to reduce memory usage
    # mdb.Close()  # OEMolDatabase doesn't have a Close() method in this version
    # db.Close()   # OEShapeDatabase doesn't have a Close() method in this version
    # Release references to allow garbage collection instead
    db = None
    mdb = None
    gc.collect()
    
    return outfname

class RocsScorer(Scorer):
    """
    A minimal viable scorer for ROCS that computes shape similarity scores 
    for a list of molecules against a query molecule using OEFlipper for isomer enumeration.
    """
    
    def __init__(self, sq_model_path=None, query_file=None, experiment_name="rocs_experiment", 
                 score_type="TanimotoCombo", use_gpu=False, max_isomers=4, max_rot_bonds=15, 
                 max_heavy_atoms=45, max_conformers=10, cpu_processes=None,
                 top_n_models=None, parallel_execution=True):
        """
        Initialize the ROCS Scorer.

        Parameters
        ----------
        sq_model_path : str or List[str], optional
            Path to the ROCS query file(s) (.sq file). Alternative to query_file.
        query_file : str or List[str], optional
            Path to the ROCS query file(s) (.sq or molecule file). Alternative to sq_model_path.
        experiment_name : str, optional
            Name of the experiment for file naming (default: "rocs_experiment").
        score_type : str, optional
            Type of score to extract (e.g., "ShapeTanimoto", "ColorTanimoto", "TanimotoCombo").
        use_gpu : bool, optional
            Whether to use GPU acceleration if available (default: False).
        max_isomers : int, optional
            Maximum number of isomers to enumerate per molecule (default: 4).
        max_rot_bonds : int, optional
            Maximum number of rotatable bonds to consider (default: 15).
        max_heavy_atoms : int, optional
            Maximum number of heavy atoms to process (default: 45).
        max_conformers : int, optional
            Maximum number of conformers to generate per molecule (default: 10).
        cpu_processes : int, optional
            Number of CPU processes to use if not using GPU. If None, will automatically
            determine based on system resources.
        top_n_models : int, optional
            If specified and using multiple models, only use the top N performing models.
        parallel_execution : bool, optional
            Whether to use parallel execution for multiple models (default: True).
        """
        super().__init__()
        
        # Handle both parameter options for the query files
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
            except oechem.OELicenseError as e:
                print(f"OpenEye license error: {e}")
                raise
            except Exception as e:
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

    def getScores(self, mols, frags=None):
        """
        Compute ROCS similarity scores for a list of molecules.

        Parameters
        ----------
        mols : List[str] or List[RDKit.Mol]
            A list of SMILES strings or RDKit molecule objects representing molecules.
        frags : List[str], optional
            A list of fragments (not used in this scorer).

        Returns
        -------
        scores : np.ndarray
            An array of similarity scores for the input molecules.
        """
        if not mols:
            return np.array([])
            
        # Get active models to use for scoring
        active_models = self.qfnames
        if self.model_selector and self.top_n_models:
            active_models = self.model_selector.get_active_models(self.top_n_models)
            
        # If we only have one model, use the original scoring path
        if len(active_models) == 1:
            return self._score_with_single_model(mols, active_models[0])
        
        # For multiple models, take the maximum score across all models
        if self.parallel_execution and len(active_models) > 1:
            max_scores = self._score_parallel_models(mols, active_models)
        else:
            max_scores = self._score_sequential_models(mols, active_models)
            
        # Update model performance metrics if we're using adaptive selection
        if self.model_selector:
            # Use a small subset of molecules for performance tracking
            sample_size = min(50, len(mols))
            if sample_size > 0:
                sample_mols = mols[:sample_size]
                model_scores = {}
                
                # Score each model on the sample to track performance
                for model in active_models:
                    model_scores[model] = float(np.mean(self._score_with_single_model(sample_mols, model)))
                
                # Update model selector with new performance data
                self.model_selector.update_model_scores(model_scores)
                
        return max_scores
    
    def _score_parallel_models(self, mols, models):
        """Score molecules with multiple models in parallel, taking the maximum score."""
        from concurrent.futures import ThreadPoolExecutor
        
        # Initialize with zeros - we'll take max scores across models
        max_scores = np.zeros(len(mols))
        
        # Use smaller subset of threads for model parallelism
        max_model_workers = min(len(models), os.cpu_count() or 4) 
        if self.use_gpu:
            # For GPU, avoid oversubscription - use single worker
            max_model_workers = 1
            
        # Create a thread pool to process models in parallel
        with ThreadPoolExecutor(max_workers=max_model_workers) as executor:
            future_to_model = {
                executor.submit(self._score_with_single_model, mols, model): model 
                for model in models
            }
            
            # Process results as they complete
            for future in future_to_model:
                try:
                    model_scores = future.result()
                    # Take element-wise maximum
                    max_scores = np.maximum(max_scores, model_scores)
                except Exception as e:
                    model = future_to_model[future]
                    print(f"Error with model {model}: {e}")
                    
        return max_scores
    
    def _score_sequential_models(self, mols, models):
        """Score molecules with multiple models sequentially, taking the maximum score."""
        # Initialize with zeros - we'll take max scores across models
        max_scores = np.zeros(len(mols))
        
        # Process each model sequentially
        for model in models:
            try:
                current_scores = self._score_with_single_model(mols, model)
                # Take element-wise maximum
                max_scores = np.maximum(max_scores, current_scores)
            except Exception as e:
                print(f"Error with model {model}: {e}")
            
            # Force garbage collection between models to free memory
            gc.collect()
            
        return max_scores
        
    def _score_with_single_model(self, mols, model_path):
        """
        Compute ROCS similarity scores for a list of molecules using a single model.
        This is the original implementation refactored to support multi-model scoring.
        """
        if not mols:
            return np.array([])

        # Check if input contains RDKit molecules and convert to SMILES if needed
        import_rdkit = False
        for mol in mols:
            if mol is not None and not isinstance(mol, str):
                import_rdkit = True
                break
                
        if import_rdkit:
            try:
                from rdkit import Chem
                smiles = []
                for mol in mols:
                    if mol is None:
                        smiles.append(None)
                    else:
                        try:
                            smi = Chem.MolToSmiles(mol)
                            smiles.append(smi)
                        except:
                            smiles.append(None)
                mols = smiles
            except ImportError:
                print("Warning: RDKit not available. Can only process SMILES inputs.")
                return np.zeros(len(mols))

        # Initial filtering to skip invalid molecules
        filtered_mols = []
        mol_ids = []
        
        for i, smi in enumerate(mols):
            # Skip missing or empty entries
            if not smi:
                continue
                
            # Basic filtering for problematic molecules
            if isinstance(smi, str) and len(smi) > 0:
                mol = oechem.OEMol()
                if oechem.OESmilesToMol(mol, str(smi)):
                    # Check heavy atom count
                    if oechem.OECount(mol, oechem.OEIsHeavy()) <= self.max_heavy_atoms:
                        filtered_mols.append((i, smi))
                        mol_ids.append(f"molecule_{i}")
        
        if not filtered_mols:
            return np.zeros(len(mols))

        # Create temporary directory for all files
        temp_dir = tempfile.mkdtemp(prefix="rocs_")
        
        try:
            # Create isomers file
            isomers_file = os.path.join(temp_dir, "isomers.smi")
            
            # Generate isomers using OEFlipper
            flipper_opts = oeomega.OEFlipperOptions()
            flipper_opts.SetMaxCenters(min(4, self.max_isomers))
            
            # Write isomers to file
            with open(isomers_file, 'w') as f:
                isomer_count = 0
                
                for orig_idx, smi in filtered_mols:
                    mol = oechem.OEMol()
                    oechem.OESmilesToMol(mol, str(smi))
                    mi = f"molecule_{orig_idx}"
                    mol.SetTitle(mi)
                    
                    try:
                        mol_isomer_count = 0
                        for iso in oeomega.OEFlipper(mol, flipper_opts):
                            if mol_isomer_count >= self.max_isomers:
                                break
                            iso_smi = oechem.OEMolToSmiles(iso)
                            msid = f"{mi}+{mol_isomer_count}"
                            f.write(f"{iso_smi}\t{msid}\n")
                            isomer_count += 1
                            mol_isomer_count += 1
                    except Exception as e:
                        print(f"Error generating isomer for {smi}: {e}")
            
            if isomer_count == 0:
                return np.zeros(len(mols))

            # Generate conformers using OMEGA
            dbname = OMEGA(isomers_file, self.experiment_name, self.max_conformers)

            # Run ROCS with appropriate thread count and only the specific model
            ofname = ROCS(dbname, [model_path], self.experiment_name, self.use_gpu, self.cpu_processes)

            # Process the output CSV more efficiently
            try:
                data = pd.read_csv(ofname)
                
                # Extract original molecule index from title
                data['MolID'] = data['TITLE'].apply(lambda x: x.split('+')[0])
                
                # Create result array with zeros
                result = np.zeros(len(mols))
                
                # More efficient groupby to get max scores
                if self.score_type in data.columns:
                    max_scores = data.groupby('MolID')[self.score_type].max()
                    
                    # Update result array directly without concat
                    for mol_id, score in max_scores.items():
                        if mol_id.startswith('molecule_'):
                            idx = int(mol_id.split('_')[1])
                            result[idx] = score
                
                return result
            except Exception as e:
                print(f"Error processing ROCS results: {e}")
                return np.zeros(len(mols))
            
        except Exception as e:
            print(f"Error in ROCS scoring: {e}")
            return np.zeros(len(mols))
            
        finally:
            # Clean up all temporary files
            for file_path in [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)]:
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
            
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"Error removing directory {temp_dir}: {e}")
                
            # Force cleanup
            gc.collect()

    def getKey(self):
        """
        Return the identifier key for this scorer.

        Returns
        -------
        str
            The key identifier "ROCS".
        """
        return "ROCS"
