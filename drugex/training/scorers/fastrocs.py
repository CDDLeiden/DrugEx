#!/usr/bin/env python3
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
FastROCS‑based scorer used by DrugEx‑ROCS.

Key features:
- Highly optimized implementation for maximum performance
- Advanced memory management techniques for large-scale processing
- Multi-threading and process pool support for CPU parallelization
- GPU acceleration with optimized data handling
- Resource-aware batch sizing and efficient caching
- Suitable for production environments and high-throughput screening
- Designed to handle thousands of molecules reliably

This implementation prioritizes performance over simplicity and is 
recommended for production environments, large molecule libraries,
virtual screening pipelines, and when maximum speed is required.
"""

from __future__ import annotations
import os, tempfile, multiprocessing as mp
import sys, time, gc, threading
from functools import partial, lru_cache
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any, Set
import warnings

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import numpy as np
from openeye import oechem, oeomega, oeshape, oefastrocs, oeff
from drugex.training.scorers.interfaces import Scorer

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# ------------------------------------------------------------------------------
# Global configuration and caching
# ------------------------------------------------------------------------------

# Global flag to track memory pool initialization
_OE_MEMORY_POOL_INITIALIZED = False

# Initialize memory pool once at module import time
def _initialize_oe_memory_pool():
    """Initialize OpenEye memory pool only once at module import time."""
    global _OE_MEMORY_POOL_INITIALIZED
    
    # Check if already initialized in this process via environment variable
    if os.environ.get("OE_MEMORY_POOL_INITIALIZED") == "true":
        _OE_MEMORY_POOL_INITIALIZED = True
        return
        
    if not _OE_MEMORY_POOL_INITIALIZED:
        try:
            oechem.OESetMemPoolMode(oechem.OEMemPoolMode_System)  # https://docs.eyesopen.com/toolkits/python/oechemtk/multithreading.html
            _OE_MEMORY_POOL_INITIALIZED = True
            os.environ["OE_MEMORY_POOL_INITIALIZED"] = "true"
            print("OpenEye memory pool initialized in fastrocs module")
        except Exception as e:
            print(f"Warning: Failed to set memory pool mode: {e}")
# Initialize at module import time
_initialize_oe_memory_pool()

# Create persistent cache directories
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "fastrocs_cache")
_DB_CACHE_DIR = os.path.join(_CACHE_DIR, "databases")
_CONF_CACHE_DIR = os.path.join(_CACHE_DIR, "conformers")

# Create cache directories if they don't exist
for d in [_CACHE_DIR, _DB_CACHE_DIR, _CONF_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# Adaptive batch sizing parameters
_MIN_BATCH_SIZE = 10
_MAX_BATCH_SIZE = 200
_TARGET_MEMORY_PER_WORKER = 1.5  # GB

# ------------------------------------------------------------------------------
#  Generic helpers
# ------------------------------------------------------------------------------

def _get_memory_info():
    """Get system memory information."""
    if PSUTIL_AVAILABLE:
        vm = psutil.virtual_memory()
        return vm.total / (1024**3), vm.available / (1024**3)  # Total and available in GB
    else:
        # Fallback to conservative estimate
        return 8.0, 4.0

def _calculate_optimal_workers(suggested_workers=None):
    """Calculate optimal number of worker processes based on system resources."""
    if PSUTIL_AVAILABLE:
        total_memory, available_memory = _get_memory_info()
        cpu_count = os.cpu_count() or 4
        
        # Leave at least 2GB or 25% for the system, whichever is larger
        reserved_memory = max(2.0, total_memory * 0.25)
        usable_memory = max(0.5, available_memory - reserved_memory)
        
        # Calculate workers based on memory constraints - more conservative
        memory_workers = max(1, int(usable_memory / (_TARGET_MEMORY_PER_WORKER * 1.5)))
        
        # Calculate workers based on CPU - leave at least 2 cores free
        cpu_workers = max(1, min(2, cpu_count - 2))
        
        # Use the minimum of memory-based and CPU-based calculations
        optimal = min(memory_workers, cpu_workers)
        
        # Honor user suggestion if provided, but don't exceed system capacity
        if suggested_workers is not None:
            return min(suggested_workers, optimal)
        return optimal
    else:
        # Fallback without psutil
        cpu_count = os.cpu_count() or 4
        return suggested_workers if suggested_workers is not None else max(1, min(2, cpu_count - 2))

def _get_file_hash(filepath):
    """Generate a simple hash for a file to use as cache key."""
    try:
        stat = os.stat(filepath)
        return f"{os.path.basename(filepath)}_{stat.st_size}_{int(stat.st_mtime)}"
    except (OSError, IOError):
        return os.path.basename(filepath)

@contextmanager
def _tmpdir(prefix="fastrocs_", use_cache=False, cache_key=None):
    """Enhanced temporary directory manager with optional caching."""
    if use_cache and cache_key:
        # Use a persistent directory in the cache
        path = os.path.join(_CACHE_DIR, f"{prefix}_{cache_key}")
        os.makedirs(path, exist_ok=True)
        try:
            yield path
        finally:
            pass  # Don't clean up cached directories
    else:
        # Use a standard temporary directory
        path = tempfile.mkdtemp(prefix=prefix)
        try:
            yield path
        finally:  # best‑effort cleanup
            if os.path.exists(path):
                for root, _, files in os.walk(path, topdown=False):
                    for f in files:
                        try: 
                            file_path = os.path.join(root, f)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        except (OSError, IOError) as e: 
                            # Just log errors but don't raise
                            print(f"Warning: Failed to remove temp file {f}: {e}")
                try: 
                    os.rmdir(path)
                except (OSError, IOError) as e:
                    print(f"Warning: Failed to remove temp directory {path}: {e}")


# ------------------------------------------------------------------------------
#  Per‑molecule utilities
# ------------------------------------------------------------------------------

# Cache for SMILES validation, significantly speeds up repeated checks
@lru_cache(maxsize=1000)
def _is_valid_smiles(smiles: str) -> bool:
    """Cached SMILES validation check."""
    if not smiles or not isinstance(smiles, str) or len(smiles) < 2:
        return False
        
    mol = oechem.OEMol()
    return bool(oechem.OESmilesToMol(mol, smiles)) # https://docs.eyesopen.com/toolkits/python/oechemtk/OEChemFunctions/OESmilesToMol.html

def filter_molecules(smiles_list: List[str], max_rot: int = 15, max_heavy: int = 45) -> List[Tuple[str, bool]]:
    """
    Filter molecules for FastROCS processing with relaxed criteria.
    Returns list of (smiles, is_valid) tuples.
    """
    # For small lists, process directly
    if len(smiles_list) <= 100:
        return _filter_molecules_chunk(smiles_list, max_rot, max_heavy)
        
    # For larger lists, use parallel processing with thread pool
    chunk_size = 50
    chunks = [smiles_list[i:i+chunk_size] for i in range(0, len(smiles_list), chunk_size)]
    
    results = []
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as executor:  # https://docs.eyesopen.com/toolkits/python/oechemtk/multithreading.html
        chunk_results = list(executor.map(
            lambda chunk: _filter_molecules_chunk(chunk, max_rot, max_heavy), chunks
        ))
        
    # Flatten results
    for cr in chunk_results:
        results.extend(cr)
            
    return results

def _filter_molecules_chunk(smiles_list: List[str], max_rot: int, max_heavy: int) -> List[Tuple[str, bool]]:
    """
    Process a chunk of molecules for filtering with more permissive criteria.
    accept more molecules.
    """
    results = []
    
    for smi in smiles_list:
        # Quick check with cached validation
        if not _is_valid_smiles(smi):
            results.append((smi, False))
            continue
            
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smi)  # https://docs.eyesopen.com/toolkits/python/oechemtk/OEChemFunctions/OESmilesToMol.html
        
        # Relaxed filtering - only check the most essential criteria
        rotatable_bonds = oechem.OECount(mol, oechem.OEIsRotor())  # https://docs.eyesopen.com/toolkits/python/oechemtk/predicates.html
        heavy_atoms = oechem.OECount(mol, oechem.OEIsHeavy())      # https://docs.eyesopen.com/toolkits/python/oechemtk/predicates.html
        
        if rotatable_bonds > max_rot or heavy_atoms > max_heavy:
            results.append((smi, False))
            continue
            
        # Basic structural check - molecule should have at least a few atoms
        atom_count = mol.NumAtoms()
        if atom_count < 3:
            results.append((smi, False))
            continue
        
        results.append((smi, True))
            
    return results

# Cached flipper options to avoid recreating them for each molecule
@lru_cache(maxsize=10)
def _get_flipper_options(max_centers=4):
    """Cached flipper options for isomer enumeration."""
    opts = oeomega.OEFlipperOptions()  # https://docs.eyesopen.com/toolkits/python/omegatk/OEConfGenClasses/OEFlipperOptions.html
    opts.SetMaxCenters(max_centers)
    return opts

def _enumerate_isomers(mol: oechem.OEMol, max_centers=4, max_iso=4):
    """Generate isomers for a molecule, optimized with cached options."""
    opts = _get_flipper_options(max_centers)
    for i, conf in enumerate(oeomega.OEFlipper(mol, opts)):  # https://docs.eyesopen.com/toolkits/python/_downloads/6c64de11ed55cc28e5f3279d66f9657b/stereo_and_torsion.py
        if i == max_iso:                                       # https://docs.eyesopen.com/toolkits/python/omegatk/omegaexamples.html
            break
        iso = oechem.OEMol(conf)
        iso.SetTitle(f"{mol.GetTitle()}+{i}")
        yield iso

# Cached omega options for conformer generation
@lru_cache(maxsize=10)
def _get_omega_options(use_gpu: bool, max_confs: int = 10):
    """Get cached omega options for conformer generation."""
    omegaOpts = oeomega.OEOmegaOptions()   # https://docs.eyesopen.com/toolkits/python/omegatk/OEConfGenClasses/OEOmega.html
    omegaOpts.SetMaxConfs(max_confs)       # https://docs.eyesopen.com/toolkits/python/omegatk/OEConfGenClasses/OEOmegaOptions.html               
    
    # Configure GPU mode for TorDrive
    try:
        if use_gpu and oeomega.OEOmegaIsGPUReady():
            # Enable GPU mode
            omegaOpts.GetTorDriveOptions().SetUseGPU(True)
            # For GPU compatibility, use recommended force field
            from openeye import oeff
            omegaOpts.GetTorDriveOptions().SetForceField(oeff.OEMMFFSheffieldFFType_MMFF94Smod_NOESTAT)
            # Disable hydrogen sampling for GPU compatibility
            omegaOpts.GetMolBuilderOptions().SetSampleHydrogens(False)  # https://docs.eyesopen.com/toolkits/python/omegatk/omegagpuomega.html
            print("Omega GPU mode enabled for conformer generation")
        else:
            omegaOpts.GetTorDriveOptions().SetUseGPU(False)
    except Exception as e:
        print(f"Warning: Error configuring Omega GPU mode: {e}")
        omegaOpts.GetTorDriveOptions().SetUseGPU(False)
    
    # Common settings for both CPU and GPU modes
    omegaOpts.SetStrictStereo(False)    # https://docs.eyesopen.com/toolkits/python/omegatk/OEConfGenClasses/OEOmegaOptions.html
    omegaOpts.SetFromCT(True)           # https://docs.eyesopen.com/toolkits/python/omegatk/OEConfGenClasses/OEMolBuilderOptions.html?highlight=setfromct
    omegaOpts.SetMaxConfs(max_confs)    # https://docs.eyesopen.com/toolkits/python/omegatk/omegaexamples.html
    
    return omegaOpts


# ShapeDatabase management for enhanced performance
class ShapeDatabaseCache:
    """Class to manage persistent shape databases with efficient caching."""
    
    def __init__(self):
        self.databases = {}
        self.lock = threading.RLock()
        
    def get_or_create_database(self, sq_model_path: str, use_gpu: bool) -> Tuple[oefastrocs.OEShapeDatabase, oeshape.OEShapeQuery, oefastrocs.OEShapeDatabaseOptions]:
        """
        Get or create a shape database for the given query.
        Returns tuple of (database, query, options).
        """
        with self.lock:
            key = f"{_get_file_hash(sq_model_path)}_{use_gpu}"
            if key in self.databases:
                db, query, opts = self.databases[key]
                # Check if we have a valid cached entry
                return db, query, opts
            
            # Create query first - needed for database preparation
            query = oeshape.OEShapeQuery()
            # Ensure sq_model_path is a single file path string, not a list
            if isinstance(sq_model_path, (list, tuple)):
                # This should never happen, but just in case
                model_path = sq_model_path[0]
                print(f"Warning: Expected single query file, got list. Using first: {model_path}")
            else:
                model_path = sq_model_path
                
            if not oeshape.OEReadShapeQuery(model_path, query):
                raise ValueError(f"Invalid shape query file: {model_path}")
            
            # Create options with correct mode setting
            opts = oefastrocs.OEShapeDatabaseOptions()   # https://docs.eyesopen.com/toolkits/python/fastrocstk/OEFastROCSClasses/OEShapeDatabaseOptions.html
            if use_gpu:
                # Use GPU mode if available and explicitly set FastROCS mode
                opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_FastROCS)   # https://docs.eyesopen.com/toolkits/python/fastrocstk/OEFastROCSClasses/OEShapeDatabaseOptions.html#OEFastROCS::OEShapeDatabaseOptions::SetFastROCSMode
                print("FastROCS GPU mode enabled for shape queries")
            else:
                # CPU mode (ROCS) - explicitly set
                opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_ROCS)
            
            # Create new database
            db = oefastrocs.OEShapeDatabase()  # https://docs.eyesopen.com/toolkits/python/fastrocstk/OEFastROCSClasses/OEShapeDatabase.html
            
            # Set thread count appropriately
            if use_gpu:
                # GPU mode - single thread is optimal
                db.SetNumOpenThreads(1)
            else:
                # CPU mode - single thread per worker to avoid contention
                db.SetNumOpenThreads(1)
            
            # Store in cache
            self.databases[key] = (db, query, opts)
            return db, query, opts
            
    def close_all(self):
        """Release all database resources"""
        with self.lock:
            # Don't call db.Close() - just clear references to allow GC to handle cleanup
            self.databases.clear()
            gc.collect()

# Global database cache
_SHAPE_DB_CACHE = ShapeDatabaseCache()

def _prepare_molecules_for_scoring(smiles_list: List[str], idxs: List[int], 
                                max_iso: int, max_rot: int, max_heavy: int, 
                                use_gpu: bool, max_confs: int = 10) -> Tuple[List[oechem.OEMol], Dict[str, int]]:
    """
    Prepare molecules for scoring by filtering, generating isomers and conformers.
    Returns list of conformers and title-to-index mapping.
    """
    # Apply simplified filtering
    filtered_data = []
    filter_results = filter_molecules(smiles_list, max_rot, max_heavy)
    
    for (smi, is_valid), idx in zip(filter_results, idxs):
        if is_valid:
            filtered_data.append((smi, idx))
    
    if not filtered_data:
        return [], {}
    
    title2parent: Dict[str, int] = {}
    isomers: List[oechem.OEMol] = []

    # Check if GPU is available for Omega
    omega_gpu = use_gpu
    if use_gpu:
        try:
            omega_gpu = oeomega.OEOmegaIsGPUReady()  # https://docs.eyesopen.com/toolkits/python/omegatk/OEConfGenFunctions/OEOmegaIsGPUReady.html
            if not omega_gpu:
                print("Warning: GPU requested but Omega GPU is not ready, using CPU for conformer generation")
        except Exception as e:
            print(f"Warning: Error checking Omega GPU status, using CPU: {e}")
            omega_gpu = False
    
    # Use cached conformer generation options
    omega = oeomega.OEOmega()
    omega.SetOptions(_get_omega_options(omega_gpu, max_confs))
    omega.SetMaxConfs(max_confs)   # https://docs.eyesopen.com/toolkits/python/omegatk/omegaexamples.html
    
    # Process sequentially for better stability
    for s, idx in filtered_data:
        result = _generate_conformers(s, str(idx), omega, max_iso)
        if result:
            mol_title2parent, mol_isomers = result
            title2parent.update(mol_title2parent)
            isomers.extend(mol_isomers)

    return isomers, title2parent

def _generate_conformers(smiles: str, idx: str, omega: oeomega.OEOmega, max_iso: int) -> Tuple[Dict[str, int], List[oechem.OEMol]]:
    """Generate conformers for a single molecule. Used for parallel conformer generation."""
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles)  # https://docs.eyesopen.com/toolkits/python/oechemtk/OEChemFunctions/OESmilesToMol.html
    mol.SetTitle(idx)
    
    title2parent = {}
    isomers = []
    
    for iso in _enumerate_isomers(mol, max_iso=max_iso):
        omega(iso)
        for conf in iso.GetConfs():
            confmol = oechem.OEMol(conf)
            title2parent[confmol.GetTitle()] = int(idx)
            isomers.append(confmol)
            
    return title2parent, isomers

def _score_molecules_with_database(isomers: List[oechem.OEMol], title2parent: Dict[str, int],
                                sq_model: str, use_gpu: bool) -> Dict[int, float]:
    """Score molecules using a cached or newly created database."""
    if not isomers:
        return {}
        
    # Get or create shape database, query, and options
    try:
        db, query, opts = _SHAPE_DB_CACHE.get_or_create_database(sq_model, use_gpu)
    except oechem.OELicenseError as e:  # pylint: disable=no-member
        print(f"OpenEye license error: {e}")
        return {}
    except Exception as e:
        print(f"Error creating shape database: {e}")
        return {}
    
    # Prepare the molecules in a temporary database
    scores: Dict[int, float] = {}
    
    # Use cache directory for database if it's CPU mode (more reusable)
    with _tmpdir(prefix="rocs_mols", use_cache=not use_gpu, 
                cache_key=_get_file_hash(sq_model) if not use_gpu else None) as td:
        # Use our optimized database preparation function
        database_path = os.path.join(td, "confs.oeb")
        mdb = _prepare_molecule_database(isomers, database_path, use_gpu)
        if not mdb:
            print(f"Error: Could not prepare molecule database")
            return {}
        
        # Create a fresh database for each batch
        fresh_db = oefastrocs.OEShapeDatabase()   # https://docs.eyesopen.com/toolkits/python/fastrocstk/tutorials/Tutorial_2_Database_Preparation/database_prep.html
        # Configure the database properly
        fresh_db.SetNumOpenThreads(1)  # Use conservative thread count
        
        # Open shape database with molecule database
        try:
            if not fresh_db.Open(mdb):
                print("Error: Could not open shape database with molecule database")
                return {}
            
            # Process scores - use the optimized API
            for sc in fresh_db.GetSortedScores(query, opts):
                mol_idx = sc.GetMolIdx()
                # Get molecule title directly from database for better performance
                mol_title = mdb.GetTitle(mol_idx)
                parent = title2parent.get(mol_title)
                if parent is not None:
                    tc = sc.GetTanimotoCombo()
                    scores[parent] = max(tc, scores.get(parent, 0.0))
        except oechem.OELicenseError as e:  # pylint: disable=no-member
            print(f"OpenEye license error during scoring: {e}")
        except Exception as e:
            print(f"Error during molecule scoring: {e}")
        finally:
            # Clean up resources by setting references to None
            # This allows Python's garbage collector to free the memory
            fresh_db = None
            mdb = None
            gc.collect()
        
    return scores

def _score_batch(batch: Tuple[List[str], List[int]],
                sq_model: str,
                max_iso: int,
                max_rot: int,
                max_heavy: int,
                use_gpu: bool = True,
                max_confs: int = 10,
                worker_id: int = None) -> Dict[int, float]:
    """
    Process and score a batch of molecules.
    Combines filtering, conformer generation, and scoring in one efficient function.
    Optimized with better memory management.
    """
    smiles, idxs = batch
    
    # Initialize worker environment
    _init_worker(worker_id)
    
    # Check GPU availability if requested
    if use_gpu:
        try:
            is_gpu_ready = oefastrocs.OEFastROCSIsGPUReady()
            if not is_gpu_ready:
                print("Warning: GPU requested but FastROCS GPU is not ready, falling back to CPU")
                use_gpu = False
        except Exception as e:
            print(f"Warning: Error checking GPU status, falling back to CPU: {e}")
            use_gpu = False
    
    # Prepare molecules (filter, generate conformers)
    isomers, title2parent = _prepare_molecules_for_scoring(
        smiles, idxs, max_iso, max_rot, max_heavy, use_gpu, max_confs
    )
    
    if not isomers:
        return {}
    
    # Score molecules
    try:
        scores = _score_molecules_with_database(isomers, title2parent, sq_model, use_gpu)
    except Exception as e:
        print(f"Error in scoring batch: {e}")
        import traceback
        traceback.print_exc()
        scores = {}
    finally:
        # Clean up to reduce memory usage - critical for reliable operation
        isomers.clear()
        title2parent.clear()
        gc.collect()
    
    return scores


# ------------------------------------------------------------------------------
#  Adaptive model selection
# ------------------------------------------------------------------------------

class AdaptiveModelSelector:
    """
    Class to adaptively select top-performing shape models during training.
    Tracks model performance and provides a mechanism to focus on the best models.
    """
    
    def __init__(self, model_paths: List[str]):
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
    
    def update_model_scores(self, new_scores: Dict[str, float]):
        """
        Update model scores with new data using exponential moving average.
        
        Parameters
        ----------
        new_scores : Dict[str, float]
            Dictionary mapping model paths to their average scores
        """
        alpha = 0.3  # Smoothing factor - higher means more weight on recent scores
        
        for model, score in new_scores.items():
            if model in self.model_scores:
                # Update with exponential moving average
                old_score = self.model_scores[model]
                self.model_scores[model] = alpha * score + (1 - alpha) * old_score
                # Increment usage counter
                self.usage_counts[model] += 1
        
        # Store history for potential analysis
        self.history.append(self.model_scores.copy())
        
        # Keep history size manageable
        if len(self.history) > 20:
            self.history.pop(0)
    
    def get_active_models(self, top_n: int = None) -> List[str]:
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
    
    def get_model_stats(self) -> Dict[str, Dict[str, float]]:
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

# ------------------------------------------------------------------------------
#  Scorer class
# ------------------------------------------------------------------------------

class OpenEyeScorer(Scorer):
    """
    A FastROCS-based scorer with optimized performance for both GPU and CPU modes.
    Compatible with the original DrugEx scorer interface.
    """

    def __init__(self,
                sq_model_path: str | List[str],
                use_gpu: bool = True,
                max_isomers: int = 4,
                max_rot_bonds: int = 10,
                max_heavy_atoms: int = 30,
                max_conformers: int = 10,
                cpu_processes: int | None = None,
                top_n_models: int | None = None,
                parallel_execution: bool = True):
        """
        Initialize the OpenEye FastROCS scorer.

        Parameters
        ----------
        sq_model_path : str or List[str]
            Path to the ROCS query file (.sq file) or list of paths for multiple models
        use_gpu : bool, optional
            Whether to use GPU acceleration if available (default: True)
        max_isomers : int, optional
            Maximum number of isomers to enumerate per molecule (default: 4)
        max_rot_bonds : int, optional
            Maximum number of rotatable bonds to consider (default: 10)
        max_heavy_atoms : int, optional
            Maximum number of heavy atoms to process (default: 30)
        max_conformers : int, optional
            Maximum number of conformers to generate per molecule (default: 10)
        cpu_processes : int | None, optional
            Number of CPU processes to use if not using GPU. If None, will use
            available CPU cores minus 2 (to leave resources for the system).
            Ignored when GPU mode is active.
        top_n_models : int | None, optional
            If specified and using multiple models, only use the top N performing models
        parallel_execution : bool, optional
            Whether to use parallel execution for multiple models (default: True)
        """
        # Handle both single model or list of models
        if isinstance(sq_model_path, (list, tuple)):
            self.sq_models = list(sq_model_path)
        else:
            self.sq_models = [sq_model_path]
            
        # Validate all model files
        for sq_path in self.sq_models:
            if not os.path.isfile(sq_path):
                raise FileNotFoundError(sq_path)
                
            # Validate each shape query file
            try:
                query = oeshape.OEShapeQuery()
                if not oeshape.OEReadShapeQuery(sq_path, query):
                    raise ValueError(f"Invalid shape query file: {sq_path}")
            except Exception as e:
                print(f"Warning: Error validating query file {sq_path}: {e}")
                # Continue with other files even if one fails

        # Memory pool initialization is now handled at the module level
        # to prevent duplicate calls across different parts of the program

        # Set up model selection
        self.top_n_models = top_n_models
        self.parallel_execution = parallel_execution
        self.model_selector = None
        
        # Initialize adaptive model selector if using multiple models and top_n_models is specified
        if len(self.sq_models) > 1 and top_n_models is not None:
            self.model_selector = AdaptiveModelSelector(self.sq_models)
            
        # Store the primary model for compatibility with older code
        self.sq_model = self.sq_models[0]
        
        # Store other parameters
        self.max_iso = max_isomers
        self.max_rot = max_rot_bonds
        self.max_heavy = max_heavy_atoms
        self.max_confs = max_conformers

        # ------------------------------------------------------------------
        #   Device selection
        # ------------------------------------------------------------------
        gpu_ready = False
        try:
            gpu_ready = use_gpu and oefastrocs.OEFastROCSIsGPUReady()
        except ImportError:
            print("Warning: FastROCS GPU not available")
            
        self.use_gpu = gpu_ready
        
        # Process count management
        if gpu_ready:
            self.cpu_procs = 0
            print("FastROCS GPU mode   : ON  (single process)")
        else:
            # Process count management for CPU mode
            if cpu_processes is not None:
                # User suggestion provided, pass it to the calculator
                self.cpu_procs = _calculate_optimal_workers(suggested_workers=cpu_processes)
            else:
                # No user suggestion, let the calculator determine based on resources
                self.cpu_procs = _calculate_optimal_workers()
            print(f"FastROCS CPU mode   : {self.cpu_procs} spawn workers")

        # Validate each query file and prepare
        try:
            # Create database options with correct mode
            opts = oefastrocs.OEShapeDatabaseOptions()
            if self.use_gpu:
                opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_FastROCS)
            else:
                opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_ROCS)
                
            # Validate the primary model used for compatibility with older code
            primary_query = oeshape.OEShapeQuery()  # https://docs.eyesopen.com/toolkits/python/shapetk/OEShapeClasses/OEShapeQuery.html
            if not oeshape.OEReadShapeQuery(self.sq_model, primary_query):   # https://docs.eyesopen.com/toolkits/python/shapetk/shape_examples.html#overlap-with-shape-query
                raise ValueError(f"Invalid shape query file: {self.sq_model}")
        except Exception as e:
            print(f"Warning: Error initializing query: {e}")

        os.environ["OE_SILENT"] = "true"
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)
        
    def __del__(self):
        """Proper cleanup of resources when the scorer is deleted"""
        try:
            # Clear any specific resources
            _SHAPE_DB_CACHE.close_all()
            # Force garbage collection to clean up any remaining handles
            gc.collect()
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  Public scoring methods
    # ------------------------------------------------------------------
    def getScores(self, smiles: List[str]) -> np.ndarray:
        """Pipeline still calls this explicitly."""
        return self._score(smiles)

    # DrugEx explorers call the object itself
    def __call__(self, mols) -> np.ndarray:
        # accept OEMol / RDKit / SMILES
        smiles = []
        for m in mols:
            if isinstance(m, str):
                smiles.append(m)
            elif isinstance(m, oechem.OEMol):
                smiles.append(oechem.OECreateSmiString(m))
            elif RDKIT_AVAILABLE and hasattr(m, "GetNumAtoms"):
                smiles.append(Chem.MolToSmiles(m))  # pylint: disable=no-member
            else:
                smiles.append("")
        return self._score(smiles)

    # ------------------------------------------------------------------
    #  Internal dispatch
    # ------------------------------------------------------------------
    def _score(self, smiles: List[str]) -> np.ndarray:
        """
        Unified scoring method for both GPU and CPU modes.
        Optimized with better resource utilization.
        Handles multiple models by taking the maximum score per molecule.
        """
        if not smiles:
            return np.zeros(0)
            
        # Get active models to use for scoring
        active_models = self.sq_models
        if self.model_selector and self.top_n_models:
            active_models = self.model_selector.get_active_models(self.top_n_models)
            
        # If we only have one model, use the original optimal scoring path
        if len(active_models) == 1:
            return self._score_with_single_model(smiles, active_models[0])
            
        # For multiple models, we'll take the maximum score across all models
        if self.parallel_execution and len(active_models) > 1:
            max_scores = self._score_parallel_models(smiles, active_models)
        else:
            max_scores = self._score_sequential_models(smiles, active_models)
            
        # Update model performance metrics if we're using adaptive selection
        if self.model_selector:
            # Use a small subset of molecules for performance tracking
            sample_size = min(50, len(smiles))
            if sample_size > 0:
                sample_smiles = smiles[:sample_size]
                model_scores = {}
                
                # Score each model on the sample to track performance
                for model in active_models:
                    model_scores[model] = float(np.mean(self._score_with_single_model(sample_smiles, model)))
                
                # Update model selector with new performance data
                self.model_selector.update_model_scores(model_scores)
            
        return max_scores
        
    def _score_parallel_models(self, smiles: List[str], models: List[str]) -> np.ndarray:
        """Score molecules with multiple models in parallel, taking the maximum score."""
        # Initialize with zeros - we'll take max scores across models
        max_scores = np.zeros(len(smiles))
        
        # Use smaller subset of threads than we would for batch processing
        max_model_workers = min(len(models), os.cpu_count() or 4) 
        if self.use_gpu:
            # For GPU, avoid oversubscription - use single worker
            max_model_workers = 1
            
        # Create a thread pool to process models in parallel
        with ThreadPoolExecutor(max_workers=max_model_workers) as executor:
            future_to_model = {
                executor.submit(self._score_with_single_model, smiles, model): model 
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
    
    def _score_sequential_models(self, smiles: List[str], models: List[str]) -> np.ndarray:
        """Score molecules with multiple models sequentially, taking the maximum score."""
        # Initialize with zeros - we'll take max scores across models
        max_scores = np.zeros(len(smiles))
        
        # Process each model sequentially
        for model in models:
            try:
                current_scores = self._score_with_single_model(smiles, model)
                # Take element-wise maximum
                max_scores = np.maximum(max_scores, current_scores)
            except Exception as e:
                print(f"Error with model {model}: {e}")
            
            # Force garbage collection between models to free memory
            gc.collect()
            
        return max_scores
        
    def _score_with_single_model(self, smiles: List[str], sq_model: str) -> np.ndarray:
        """
        Score molecules with a single model.
        Optimized with better resource utilization.
        """
        if not smiles:
            return np.zeros(0)
            
        # Adjusted batch size calculation based on mode
        if self.use_gpu:
            # GPU mode - use standard batch size
            batch_size = 30
        else:
            # CPU mode - use smaller batches based on molecule count
            if len(smiles) <= 60:
                batch_size = 15  # Smaller batches for smaller sets
            else:
                batch_size = 10  # Very small batches for larger sets
        
        # Prepare batches with the optimized size
        idxs = list(range(len(smiles)))
        batches = []
        for i in range(0, len(smiles), batch_size):
            end_idx = min(i + batch_size, len(smiles))
            batches.append((smiles[i:end_idx], idxs[i:end_idx]))
            
        # For GPU mode: use single-process scoring with optimized memory handling
        if self.use_gpu:
            results = {}
            
            # Process each batch
            for i, batch in enumerate(batches):
                try:
                    batch_results = _score_batch(
                        batch, sq_model, self.max_iso, 
                        self.max_rot, self.max_heavy, True, self.max_confs
                    )
                    results.update(batch_results)
                except Exception as e:
                    print(f"Error in GPU mode batch {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Force garbage collection between batches to prevent memory growth
                gc.collect()
                
        # For CPU mode: use simplified approach for better stability
        else:
            results = {}
            
            # For small molecule sets, process sequentially to avoid multiprocessing overhead
            if len(smiles) <= 60:
                for batch in batches:
                    try:
                        batch_results = _score_batch(
                            batch, sq_model, self.max_iso, 
                            self.max_rot, self.max_heavy, False, self.max_confs
                        )
                        results.update(batch_results)
                    except Exception as e:
                        print(f"Error in sequential CPU mode: {e}")
                    # Force cleanup
                    gc.collect()
            else:
                # For larger sets, use limited multiprocessing with spawn context for stability
                ctx = mp.get_context("spawn")  # More reliable than forkserver
                
                with ProcessPoolExecutor(max_workers=self.cpu_procs,
                                         mp_context=ctx) as pool:
                    # Pass worker ID to each process
                    futures = []
                    for i, batch in enumerate(batches):
                        worker_id = i % self.cpu_procs
                        futures.append(pool.submit(
                            _score_batch,
                            batch=batch,
                            sq_model=sq_model,
                            max_iso=self.max_iso,
                            max_rot=self.max_rot,
                            max_heavy=self.max_heavy,
                            use_gpu=False,
                            max_confs=self.max_confs,
                            worker_id=worker_id
                        ))
                    
                    # Process results as they complete
                    for future in futures:
                        try:
                            batch_results = future.result()
                            results.update(batch_results)
                        except Exception as e:
                            print(f"Error in worker process: {e}")
        
        # Convert dictionary to array
        out = np.zeros(len(smiles))
        for k, v in results.items():
            if k < len(smiles):  # Make sure index is valid
                out[k] = v
            
        # Final cleanup to minimize memory usage after processing
        gc.collect()
            
        return out

    # ------------------------------------------------------------------
    def getKey(self):
        return "ROCS"

def _prepare_molecule_database(molecules: List[oechem.OEMol], output_path: str, use_gpu: bool = False) -> oechem.OEMolDatabase:
    """
    Prepare a molecule database optimized for FastROCS or ROCS processing.
    Implements best practices from OEShapeDatabasePrep.
    
    Parameters
    ----------
    molecules : List[oechem.OEMol]
        List of molecules to include in the database
    output_path : str
        Path where the database will be written
    use_gpu : bool
        Whether to optimize for GPU usage
        
    Returns
    -------
    oechem.OEMolDatabase
        The prepared molecule database
    """
    if not molecules:
        return None
        
    # Use optimal strategy for GPU vs CPU
    if use_gpu:
        print(f"Preparing {len(molecules)} molecules for GPU processing...")
    else:
        print(f"Preparing {len(molecules)} molecules for CPU processing...")
    
    # Write molecules to SDF with optimized settings
    with oechem.oemolostream() as ofs:  # https://docs.eyesopen.com/toolkits/python/oechemtk/OEChemClasses/oemolostream.html
        if use_gpu:
            # Use PRE-Compression for faster database loading
            oechem.OEPRECompress(ofs)
        
        if not ofs.open(output_path):
            print(f"Error: Could not open output file {output_path}")
            return None
            
        processed_count = 0
        
        for mol in molecules:
            # Apply proper preparation if using GPU
            if use_gpu:
                try:
                    # Prepare molecule specifically for FastROCS
                    oefastrocs.OEPrepareFastROCSMol(mol)
                    
                    # Use half-precision for better memory usage
                    half_mol = oechem.OEMol(mol, oechem.OEMCMolType_HalfFloatCartesian)  # https://docs.eyesopen.com/toolkits/python/fastrocstk/tutorials/Tutorial_2_Database_Preparation/database_prep.html
                    oechem.OEWriteMolecule(ofs, half_mol)                                # https://docs.eyesopen.com/toolkits/python/_downloads/1db5344ca79eb4d6c5de8a2eb3bd7a52/SimplePrepScript.py
                    processed_count += 1
                except Exception as e:
                    # Fall back to standard preparation
                    print(f"Warning: Could not prepare molecule for GPU: {e}")
                    oechem.OEWriteMolecule(ofs, mol)
            else:
                oechem.OEWriteMolecule(ofs, mol)
    
    if use_gpu and processed_count > 0:
        # Less verbose message, only log if there's a significant discrepancy
        if processed_count < len(molecules) * 0.9:  # Only log if more than 10% failed
            print(f"GPU preparation: {processed_count}/{len(molecules)} molecules processed")
    
    # Create and open the molecule database
    mdb = oechem.OEMolDatabase()  # https://docs.eyesopen.com/toolkits/python/oechemtk/moldatabase.html
    if not mdb.Open(output_path):  # https://docs.eyesopen.com/toolkits/python/oechemtk/OEChemClasses/OEMolDatabase.html
        return None
        
    return mdb

def _init_worker(worker_id=None):
    """Improved initializer for worker processes with memory pool configuration."""
    # Silence OpenEye warnings
    os.environ["OE_SILENT"] = "true"
    # Set OpenEye error level
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)
    
    # Explicitly disable CUDA in worker processes
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
    # Clear any remaining GPU memory references
    gc.collect()
    
    # Set CPU affinity if possible to prevent contention
    if PSUTIL_AVAILABLE and worker_id is not None:
        try:
            process = psutil.Process()
            cpu_count = psutil.cpu_count(logical=True)
            if cpu_count > 0:
                # Simple round-robin assignment of cores
                cpu_id = worker_id % cpu_count
                process.cpu_affinity([cpu_id])
        except Exception:
            pass  # Skip if not supported or failed
