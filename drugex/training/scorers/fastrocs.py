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

def _get_gpu_memory_info():
    """Get GPU memory information if available."""
    try:
        if PSUTIL_AVAILABLE:
            # Check if NVIDIA-SMI is available
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used',
                                      '--format=csv,nounits,noheader'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               universal_newlines=True, check=False)
            if result.returncode == 0:
                # Parse the output
                lines = result.stdout.strip().split('\n')
                if lines and ',' in lines[0]:
                    total, used = map(int, lines[0].split(','))
                    total_gb = total / 1024.0
                    used_gb = used / 1024.0
                    available_gb = total_gb - used_gb
                    return total_gb, available_gb
    except Exception as e:
        print(f"Warning: Error getting GPU memory info: {e}")
    
    # Fallback values if GPU memory info can't be retrieved
    return 8.0, 4.0  # Assume 8GB GPU with 4GB available

def _monitor_memory(threshold_pct=85):
    """Monitor system memory and return True if memory usage exceeds threshold."""
    if PSUTIL_AVAILABLE:
        vm = psutil.virtual_memory()
        return vm.percent > threshold_pct
    return False

def _monitor_gpu_memory(threshold_pct=85):
    """Monitor GPU memory and return True if memory usage exceeds threshold."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used',
                                  '--format=csv,nounits,noheader'], 
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                           universal_newlines=True, check=False)
        if result.returncode == 0:
            # Parse the output
            lines = result.stdout.strip().split('\n')
            if lines and ',' in lines[0]:
                total, used = map(int, lines[0].split(','))
                used_pct = (used / total) * 100
                return used_pct > threshold_pct
    except Exception:
        pass
    
    # If any issues occur, assume memory isn't under pressure
    return False

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

def _calculate_molecule_complexity(smiles):
    """Calculate complexity score for a molecule based on size and flexibility."""
    if not smiles or not isinstance(smiles, str):
        return 30  # Default complexity for invalid input
        
    try:
        mol = oechem.OEMol()
        if not oechem.OESmilesToMol(mol, smiles):
            return 30  # Default for parsing failures
            
        # Calculate complexity based on:
        # 1. Number of heavy atoms (size)
        # 2. Number of rotatable bonds (flexibility)
        # 3. Number of rings (complexity)
        heavy_atoms = oechem.OECount(mol, oechem.OEIsHeavy())
        rotatable_bonds = oechem.OECount(mol, oechem.OEIsRotor())
        rings = 0
        for ring in oechem.OEGetSSSR(mol):
            rings += 1
        
        # Weighted complexity score - higher values mean more complex
        complexity = (heavy_atoms * 1.0) + (rotatable_bonds * 2.5) + (rings * 2.0)
        return complexity
    except Exception:
        return 30  # Default complexity if calculation fails

def _calculate_optimal_batch_size(smiles_list, use_gpu):
    """Calculate optimal batch size based on molecule complexity and available memory."""
    if not smiles_list:
        return _MIN_BATCH_SIZE
    
    # Sample molecules to estimate average complexity
    sample_size = min(100, len(smiles_list))
    sample_indices = np.linspace(0, len(smiles_list)-1, sample_size, dtype=int)
    
    # Get complexity scores for the sample
    complexity_scores = []
    for idx in sample_indices:
        try:
            complexity = _calculate_molecule_complexity(smiles_list[idx])
            complexity_scores.append(complexity)
        except Exception:
            complexity_scores.append(30)  # Default if calculation fails
    
    # Calculate average complexity
    avg_complexity = sum(complexity_scores) / max(1, len(complexity_scores))
    
    # Get available memory
    if use_gpu:
        total_memory, available_memory = _get_gpu_memory_info()
        
        # For GPU, adjust batch size based on complexity and available GPU memory
        # More complex molecules need smaller batches
        base_size = int(1500 / max(1, avg_complexity / 20))
        
        # Further adjust based on available memory
        memory_factor = min(1.0, available_memory / 4.0)  # 4GB as reference point
        adjusted_size = max(_MIN_BATCH_SIZE, int(base_size * memory_factor))
        
        # Use stricter limits for GPU to avoid OOM
        return min(_MAX_BATCH_SIZE // 2, adjusted_size)
    else:
        total_memory, available_memory = _get_memory_info()
        
        # For CPU, we can use larger batches
        base_size = int(2500 / max(1, avg_complexity / 20))
        
        # Adjust based on available system memory
        memory_factor = min(1.0, available_memory / 8.0)  # 8GB as reference point
        adjusted_size = max(_MIN_BATCH_SIZE, int(base_size * memory_factor))
        
        return min(_MAX_BATCH_SIZE, adjusted_size)

def _reduce_batch_size_on_memory_pressure(current_size, use_gpu):
    """Reduce batch size when memory pressure is detected."""
    # Check current memory pressure
    if use_gpu and _monitor_gpu_memory(threshold_pct=80):
        # Significant GPU memory pressure - reduce by 50%
        return max(_MIN_BATCH_SIZE, current_size // 2)
    elif _monitor_memory(threshold_pct=85):
        # Significant system memory pressure - reduce by 40%
        return max(_MIN_BATCH_SIZE, int(current_size * 0.6))
    else:
        # Minor adjustment - reduce by 20%
        return max(_MIN_BATCH_SIZE, int(current_size * 0.8))

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

# Shared cache for multi-model optimization
class FastROCSSharedCache:
    """
    Cache for FastROCS databases to avoid redundant preparation across models.
    This significantly speeds up multi-model scoring by sharing prepared molecules.
    """
    def __init__(self):
        self.molecule_databases = {}
        self.cached_results = {}
        self.lock = threading.RLock()
        self.max_cache_size = 5  # Maximum number of cached databases
        
    def _compute_smiles_hash(self, smiles_list):
        """Generate a stable hash for a list of SMILES."""
        if not smiles_list:
            return "empty"
            
        # Use sample of SMILES for faster hashing
        sample_size = min(100, len(smiles_list))
        sample_indices = np.linspace(0, len(smiles_list)-1, sample_size, dtype=int)
        sample = [smiles_list[i] for i in sample_indices]
        
        # Create hash from concatenated SMILES
        import hashlib
        hash_input = "".join(str(s) for s in sample) + str(len(smiles_list))
        return hashlib.md5(hash_input.encode()).hexdigest()
        
    def get_or_create_molecule_database(self, smiles_list, process_func, *args, **kwargs):
        """
        Get cached molecule database or create a new one.
        
        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES to hash for cache lookup
        process_func : callable
            Function to process molecules if not in cache
        *args, **kwargs
            Arguments to pass to process_func
            
        Returns
        -------
        Tuple
            (database_path, title2parent) - Path to molecule database and mapping
        """
        smiles_hash = self._compute_smiles_hash(smiles_list)
        
        with self.lock:
            if smiles_hash in self.molecule_databases:
                print(f"Using cached molecule database for {len(smiles_list)} compounds")
                return self.molecule_databases[smiles_hash]
            
            # Create new database
            print(f"Creating shared molecule database for {len(smiles_list)} compounds")
            result = process_func(smiles_list, *args, **kwargs)
            
            # Manage cache size
            if len(self.molecule_databases) >= self.max_cache_size:
                # Remove oldest entry (first item in dict)
                if self.molecule_databases:
                    oldest_key = next(iter(self.molecule_databases))
                    del self.molecule_databases[oldest_key]
                    
            # Add to cache
            self.molecule_databases[smiles_hash] = result
            return result
            
    def get_cached_score(self, smiles_list, model_path):
        """
        Get cached scoring result if available.
        
        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES to hash for cache lookup
        model_path : str
            Path to model file for cache lookup
            
        Returns
        -------
        Dict or None
            Cached scoring results or None if not found
        """
        smiles_hash = self._compute_smiles_hash(smiles_list)
        model_hash = _get_file_hash(model_path)
        key = f"{smiles_hash}_{model_hash}"
        
        with self.lock:
            return self.cached_results.get(key)
            
    def cache_score(self, smiles_list, model_path, results):
        """
        Cache scoring results.
        
        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES to hash for cache key
        model_path : str
            Path to model file for cache key
        results : Dict
            Dictionary of scoring results to cache
        """
        smiles_hash = self._compute_smiles_hash(smiles_list)
        model_hash = _get_file_hash(model_path)
        key = f"{smiles_hash}_{model_hash}"
        
        with self.lock:
            # Limit the number of cached results
            if len(self.cached_results) >= 10:  # Cache at most 10 results
                # Remove oldest entry
                if self.cached_results:
                    oldest_key = next(iter(self.cached_results))
                    del self.cached_results[oldest_key]
                    
            self.cached_results[key] = results
            
    def clear(self):
        """Clear all caches to free memory."""
        with self.lock:
            self.molecule_databases.clear()
            self.cached_results.clear()
            gc.collect()

# Global shared cache
_SHARED_CACHE = FastROCSSharedCache()

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
            omega_gpu = oeomega.OEOmegaIsGPUReady()
            if not omega_gpu:
                print("Warning: GPU requested but Omega GPU is not ready, using CPU for conformer generation")
        except Exception as e:
            print(f"Warning: Error checking Omega GPU status, using CPU: {e}")
            omega_gpu = False
    
    # Use cached conformer generation options
    omega = oeomega.OEOmega()
    omega.SetOptions(_get_omega_options(omega_gpu, max_confs))
    omega.SetMaxConfs(max_confs)
    
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

def _safe_process_batch(batch, sq_model, max_iso, max_rot, max_heavy, use_gpu, max_confs, worker_id=None, retry_count=0):
    """
    Process a batch with memory safety mechanisms and automatic retries on memory errors.
    
    Parameters
    ----------
    batch : Tuple[List[str], List[int]]
        Batch of SMILES strings and their indices
    sq_model : str
        Path to the shape query file
    max_iso, max_rot, max_heavy, use_gpu, max_confs : 
        Standard parameters for scoring
    worker_id : int, optional
        Worker ID for parallel processing
    retry_count : int, optional
        Current retry attempt count
        
    Returns
    -------
    Dict[int, float]
        Dictionary mapping indices to scores
    """
    max_retries = 3
    smiles, idxs = batch
    
    # If empty batch, return early
    if not smiles:
        return {}
    
    try:
        # Regular batch processing
        return _score_batch(batch, sq_model, max_iso, max_rot, max_heavy, use_gpu, max_confs, worker_id)
    
    except (MemoryError, RuntimeError, oechem.OEException) as e:
        # Check if we've exceeded max retries
        if retry_count >= max_retries:
            print(f"Warning: Maximum retries exceeded for batch. Error: {e}")
            # Return empty results to avoid crashing the entire process
            return {}
            
        # Force aggressive garbage collection
        gc.collect()
        
        # Memory error handling - split batch and retry
        if len(smiles) <= _MIN_BATCH_SIZE:
            # If batch is already small, log warning and return empty results
            print(f"Warning: Memory error on minimum batch size ({len(smiles)}). Error: {e}")
            return {}
            
        print(f"Memory pressure detected during batch processing. Splitting batch of size {len(smiles)}.")
        # Split batch in half and retry each half
        mid_idx = len(smiles) // 2
        batch1 = (smiles[:mid_idx], idxs[:mid_idx])
        batch2 = (smiles[mid_idx:], idxs[mid_idx:])
        
        # Process each half with increased retry counter
        results1 = _safe_process_batch(batch1, sq_model, max_iso, max_rot, max_heavy, 
                                      use_gpu, max_confs, worker_id, retry_count + 1)
        
        # Force garbage collection between batches
        gc.collect()
        
        results2 = _safe_process_batch(batch2, sq_model, max_iso, max_rot, max_heavy, 
                                      use_gpu, max_confs, worker_id, retry_count + 1)
        
        # Combine results
        combined_results = {}
        combined_results.update(results1)
        combined_results.update(results2)
        return combined_results
        
    except Exception as e:
        # For other unexpected exceptions, log and return empty results
        print(f"Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        return {}

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
            
        # Print hardware configuration for diagnostics
        if self.use_gpu:
            total_gpu, avail_gpu = _get_gpu_memory_info()
            print(f"GPU: {avail_gpu:.1f}GB/{total_gpu:.1f}GB available")
        total_mem, avail_mem = _get_memory_info()
        print(f"Memory: {avail_mem:.1f}GB/{total_mem:.1f}GB available")
            
        # Determine optimal checkpoint size for large datasets
        # based on available memory and GPU resources
        optimal_checkpoint_size = self._calculate_optimal_checkpoint_size(len(smiles))
            
        # Large dataset handling with progressive processing
        if len(smiles) > optimal_checkpoint_size:
            return self._score_with_checkpointing(smiles, checkpoint_size=optimal_checkpoint_size)
            
        # Get active models to use for scoring
        active_models = self.sq_models
        if self.model_selector and self.top_n_models:
            active_models = self.model_selector.get_active_models(self.top_n_models)
            
        # Print model information
        print(f"Using {len(active_models)} shape models for scoring {len(smiles)} molecules")
            
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
                # Sample diverse molecules for better performance measurement
                # We use regularly spaced indices to get a representative sample
                sample_indices = np.linspace(0, len(smiles)-1, sample_size, dtype=int)
                sample_smiles = [smiles[i] for i in sample_indices]
                
                model_scores = {}
                # Score each model on the sample to track performance
                for model in active_models:
                    # Check if we have cached results first
                    cached_sample_results = _SHARED_CACHE.get_cached_score(sample_smiles, model)
                    if cached_sample_results:
                        # Use cached results
                        avg_score = sum(cached_sample_results.values()) / max(1, len(cached_sample_results))
                        model_scores[model] = float(avg_score)
                    else:
                        # Score the sample with this model
                        sample_results = self._score_with_single_model(sample_smiles, model)
                        model_scores[model] = float(np.mean(sample_results))
                
                # Update model selector with new performance data
                self.model_selector.update_model_scores(model_scores)
            
        return max_scores
        
    def _calculate_optimal_checkpoint_size(self, total_molecules):
        """
        Calculate optimal checkpoint size based on system resources.
        
        Parameters
        ----------
        total_molecules : int
            Total number of molecules to process
            
        Returns
        -------
        int
            Optimal checkpoint size
        """
        if self.use_gpu:
            # Get available GPU memory
            _, available_gpu_mem = _get_gpu_memory_info()
            
            # GPU memory-based checkpoint size calculation (conservative)
            # Based on FastROCS architecture docs: ~4M conformers per 1GB
            # Assuming ~10 conformers per molecule on average, that's ~400K molecules per GB
            # Use a conservative estimate of 100K molecules per GB
            base_checkpoint = int(100000 * available_gpu_mem / 8.0)  # Scale by available memory
        else:
            # Get available system memory
            _, available_mem = _get_memory_info()
            
            # CPU memory-based checkpoint size calculation
            base_checkpoint = int(100000 * available_mem / 16.0)  # Scale by available memory
        
        # Scale by number of models
        num_models = max(1, min(len(self.sq_models), self.top_n_models or len(self.sq_models)))
        model_adjusted = int(base_checkpoint / max(1, num_models - 0.5))
        
        # Scale if large number of molecules to process
        if total_molecules > 50000:
            # When processing very large sets, use smaller checkpoints
            # to enable more frequent garbage collection
            size_adjusted = min(model_adjusted, 10000)
        else:
            size_adjusted = min(model_adjusted, 20000)
            
        # Always stay within reasonable bounds
        return max(1000, min(size_adjusted, 20000))
        
    def _score_with_checkpointing(self, smiles: List[str], checkpoint_size: int = 10000) -> np.ndarray:
        """
        Process large SMILES lists with checkpointing to prevent memory issues.
        
        Parameters
        ----------
        smiles : List[str]
            List of SMILES strings to score
        checkpoint_size : int, optional
            Size of each checkpoint batch (default: 10000)
            
        Returns
        -------
        np.ndarray
            Array of scores for all molecules
        """
        # If list is small enough, just score directly
        if len(smiles) <= checkpoint_size:
            return self._score(smiles)
            
        print(f"Processing large dataset of {len(smiles)} molecules with checkpointing (checkpoint size: {checkpoint_size})")
        
        # Initialize results array
        results = np.zeros(len(smiles))
        
        # Get active models
        active_models = self.sq_models
        if self.model_selector and self.top_n_models:
            active_models = self.model_selector.get_active_models(self.top_n_models)
            
        # Process in chunks to manage memory
        import time
        start_time = time.time()
        total_chunks = (len(smiles) - 1) // checkpoint_size + 1
        
        for chunk_idx, start_idx in enumerate(range(0, len(smiles), checkpoint_size)):
            # Calculate end index for this chunk
            end_idx = min(start_idx + checkpoint_size, len(smiles))
            chunk_size = end_idx - start_idx
            
            elapsed = time.time() - start_time
            est_remaining = (elapsed / (chunk_idx + 1)) * (total_chunks - chunk_idx - 1) if chunk_idx > 0 else 0
            
            print(f"Processing checkpoint {chunk_idx+1}/{total_chunks}: "
                  f"molecules {start_idx}-{end_idx-1} "
                  f"(elapsed: {elapsed:.1f}s, est. remaining: {est_remaining:.1f}s)")
                  
            # Extract chunk
            chunk_smiles = smiles[start_idx:end_idx]
            
            # Release memory before processing chunk
            gc.collect()
            
            # For multiple models, take max scores across models
            try:
                if len(active_models) > 1:
                    if self.parallel_execution:
                        chunk_scores = self._score_parallel_models(chunk_smiles, active_models)
                    else:
                        chunk_scores = self._score_sequential_models(chunk_smiles, active_models)
                else:
                    # Single model case
                    chunk_scores = self._score_with_single_model(chunk_smiles, active_models[0])
                
                # Store results for this chunk
                results[start_idx:end_idx] = chunk_scores
            except Exception as e:
                print(f"Error processing checkpoint {chunk_idx+1}: {e}")
                # Continue with next checkpoint even if this one fails
                
            # Force garbage collection after processing chunk
            gc.collect()
            
            # Periodic memory check
            if _monitor_memory(threshold_pct=90) or (self.use_gpu and _monitor_gpu_memory(threshold_pct=90)):
                print("High memory pressure detected. Reducing checkpoint size.")
                checkpoint_size = max(1000, checkpoint_size // 2)
            
        total_time = time.time() - start_time
        print(f"Completed processing of {len(smiles)} molecules in {total_time:.1f}s "
              f"({len(smiles)/total_time:.1f} molecules/second)")
            
        return results

    def _score_parallel_models(self, smiles: List[str], models: List[str]) -> np.ndarray:
        """
        Score molecules with multiple models in parallel, taking the maximum score.
        Uses shared molecule database for improved performance.
        """
        if not smiles or not models:
            return np.zeros(len(smiles))
        
        # Get GPU memory status to decide on parallelism
        if self.use_gpu:
            total_gpu_mem, available_gpu_mem = _get_gpu_memory_info()
            # Use conservative thread count based on GPU memory
            max_model_workers = 1  # Default for GPU - avoid oversubscription
        else:
            # For CPU, use smaller subset of available cores for model parallelism
            total_workers = _calculate_optimal_workers_for_workload(len(smiles), False)
            max_model_workers = min(len(models), total_workers)
        
        print(f"Scoring with {len(models)} models using {max_model_workers} parallel workers")
        
        # Prepare shared molecule database once for all models
        database_path, title2parent = _SHARED_CACHE.get_or_create_molecule_database(
            smiles, 
            _prepare_shared_molecule_database,
            list(range(len(smiles))),
            self.max_iso,
            self.max_rot,
            self.max_heavy,
            self.use_gpu,
            self.max_confs
        )
        
        if not database_path or not title2parent:
            # No valid molecules to score
            return np.zeros(len(smiles))
        
        # Initialize with zeros - we'll take max scores across models
        all_results = {}
        
        # For GPU mode with multiple models, we need to be careful with parallelism
        if self.use_gpu and len(models) > 1:
            # Score models sequentially when using GPU to avoid memory issues
            for model in models:
                try:
                    # Check if we have cached results
                    cached_results = _SHARED_CACHE.get_cached_score(smiles, model)
                    if cached_results:
                        print(f"Using cached results for model: {os.path.basename(model)}")
                        model_results = cached_results
                    else:
                        # Score with this model
                        model_results = _score_with_shared_database(
                            database_path, title2parent, model, self.use_gpu
                        )
                        # Cache the results
                        _SHARED_CACHE.cache_score(smiles, model, model_results)
                    
                    # Update with max scores
                    for idx, score in model_results.items():
                        all_results[idx] = max(score, all_results.get(idx, 0.0))
                        
                    # Force garbage collection after each model
                    gc.collect()
                except Exception as e:
                    print(f"Error with model {model}: {e}")
        else:
            # Use thread pool to score with each model in parallel
            with ThreadPoolExecutor(max_workers=max_model_workers) as executor:
                futures = []
                for model in models:
                    # Check if we have cached results
                    cached_results = _SHARED_CACHE.get_cached_score(smiles, model)
                    if cached_results:
                        print(f"Using cached results for model: {os.path.basename(model)}")
                        # Use a completed future to represent cached results
                        from concurrent.futures import Future
                        future = Future()
                        future.set_result(cached_results)
                        futures.append((future, model))
                    else:
                        # Submit new scoring task
                        future = executor.submit(
                            _score_with_shared_database,
                            database_path=database_path,
                            title2parent=title2parent,
                            sq_model=model,
                            use_gpu=self.use_gpu
                        )
                        futures.append((future, model))
                
                # Process results from each model, keeping max score per molecule
                for future, model in futures:
                    try:
                        model_results = future.result()
                        # Cache results for future use
                        if not _SHARED_CACHE.get_cached_score(smiles, model):
                            _SHARED_CACHE.cache_score(smiles, model, model_results)
                        
                        # Update with max scores
                        for idx, score in model_results.items():
                            all_results[idx] = max(score, all_results.get(idx, 0.0))
                    except Exception as e:
                        print(f"Error with model {model}: {e}")
        
        # Convert dictionary to array
        max_scores = np.zeros(len(smiles))
        for idx, score in all_results.items():
            if 0 <= idx < len(smiles):
                max_scores[idx] = score
                
        # Force memory cleanup
        gc.collect()
                
        return max_scores
    
    def _score_sequential_models(self, smiles: List[str], models: List[str]) -> np.ndarray:
        """
        Score molecules with multiple models sequentially, taking the maximum score.
        Uses shared molecule database for improved performance.
        """
        if not smiles or not models:
            return np.zeros(len(smiles))
        
        # Prepare shared molecule database once for all models
        database_path, title2parent = _SHARED_CACHE.get_or_create_molecule_database(
            smiles, 
            _prepare_shared_molecule_database,
            list(range(len(smiles))),
            self.max_iso,
            self.max_rot,
            self.max_heavy,
            self.use_gpu,
            self.max_confs
        )
        
        if not database_path or not title2parent:
            # No valid molecules to score
            return np.zeros(len(smiles))
        
        # Initialize with zeros - we'll take max scores across models
        all_results = {}
        
        # Process each model sequentially
        for model in models:
            try:
                # Check if we have cached results
                cached_results = _SHARED_CACHE.get_cached_score(smiles, model)
                if cached_results:
                    print(f"Using cached results for model: {os.path.basename(model)}")
                    model_results = cached_results
                else:
                    # Score with this model
                    model_results = _score_with_shared_database(
                        database_path, title2parent, model, self.use_gpu
                    )
                    # Cache the results
                    _SHARED_CACHE.cache_score(smiles, model, model_results)
                
                # Update with max scores
                for idx, score in model_results.items():
                    all_results[idx] = max(score, all_results.get(idx, 0.0))
                    
                # Force garbage collection after each model
                gc.collect()
            except Exception as e:
                print(f"Error with model {model}: {e}")
        
        # Convert dictionary to array
        max_scores = np.zeros(len(smiles))
        for idx, score in all_results.items():
            if 0 <= idx < len(smiles):
                max_scores[idx] = score
                
        return max_scores
        
    def _score_with_single_model(self, smiles: List[str], sq_model: str) -> np.ndarray:
        """
        Score molecules with a single model.
        Optimized with adaptive batch sizing and memory-aware processing.
        """
        if not smiles:
            return np.zeros(0)
        
        # Pre-filter molecules to quickly eliminate those that are too complex
        filtered_data = _prefilter_molecules(
            smiles,
            max_rot=self.max_rot,
            max_heavy=self.max_heavy
        )
        
        # If no molecules pass filtering, return zeros
        if not filtered_data:
            return np.zeros(len(smiles))
        
        # Extract filtered smiles and their original indices
        filtered_indices = [idx for idx, _ in filtered_data]
        filtered_smiles = [smi for _, smi in filtered_data]
        
        # Calculate optimal batch size based on molecule complexity and memory
        batch_size = _calculate_optimal_batch_size(filtered_smiles, self.use_gpu)
        print(f"Using adaptive batch size: {batch_size} for {len(filtered_smiles)} molecules")
        
        # Prepare batches with the optimized size
        batches = []
        for i in range(0, len(filtered_smiles), batch_size):
            end_idx = min(i + batch_size, len(filtered_smiles))
            batches.append((filtered_smiles[i:end_idx], filtered_indices[i:end_idx]))
        
        # Process batches based on GPU or CPU mode
        results = {}
        
        if self.use_gpu:
            # GPU mode processing
            for i, batch in enumerate(batches):
                try:
                    # Check for memory pressure before processing batch
                    if _monitor_gpu_memory(threshold_pct=75) or _monitor_memory(threshold_pct=85):
                        # Reduce batch size under memory pressure
                        new_batch_size = _reduce_batch_size_on_memory_pressure(batch_size, True)
                        if new_batch_size < batch_size:
                            print(f"Memory pressure detected. Reducing batch size: {batch_size} -> {new_batch_size}")
                            # Recalculate batches with new size
                            batch_size = new_batch_size
                            batches = []
                            for j in range(0, len(filtered_smiles), batch_size):
                                end_j = min(j + batch_size, len(filtered_smiles))
                                batches.append((filtered_smiles[j:end_j], filtered_indices[j:end_j]))
                            # Restart processing with new batches
                            results = {}
                            i = -1  # Will be incremented to 0 in next iteration
                            continue
                    
                    # Process batch with memory safety
                    batch_results = _safe_process_batch(
                        batch, sq_model, self.max_iso, 
                        self.max_rot, self.max_heavy, True, self.max_confs
                    )
                    results.update(batch_results)
                    
                    # Force garbage collection between batches
                    gc.collect()
                except Exception as e:
                    print(f"Error in GPU mode batch {i+1}: {e}")
        else:
            # CPU mode processing
            # For small molecule sets, process sequentially
            if len(filtered_smiles) <= 60:
                for batch in batches:
                    try:
                        batch_results = _safe_process_batch(
                            batch, sq_model, self.max_iso, 
                            self.max_rot, self.max_heavy, False, self.max_confs
                        )
                        results.update(batch_results)
                        # Force cleanup
                        gc.collect()
                    except Exception as e:
                        print(f"Error in sequential CPU mode: {e}")
            else:
                # For larger sets, use calculated number of processes
                worker_count = min(
                    self.cpu_procs,
                    _calculate_optimal_workers_for_workload(len(filtered_smiles), False)
                )
                
                print(f"Using {worker_count} worker processes for {len(filtered_smiles)} molecules")
                
                # Use spawn context for better stability
                ctx = mp.get_context("spawn")
                
                with ProcessPoolExecutor(max_workers=worker_count,
                                        mp_context=ctx) as pool:
                    # Submit jobs to the pool
                    futures = []
                    for i, batch in enumerate(batches):
                        worker_id = i % worker_count
                        futures.append(pool.submit(
                            _safe_process_batch,
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
        
        # Final cleanup
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

def _prefilter_molecules(smiles_list: List[str], max_rot: int = 20, max_heavy: int = 50, max_complexity: int = 150) -> List[Tuple[int, str]]:
    """
    Prefilter molecules to quickly identify those that are too complex for effective scoring.
    Returns list of (index, smiles) tuples for molecules that pass the filtering.
    
    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings to filter
    max_rot : int, optional
        Maximum number of rotatable bonds (default: 20)
    max_heavy : int, optional
        Maximum number of heavy atoms (default: 50)
    max_complexity : int, optional
        Maximum complexity score (default: 150)
        
    Returns
    -------
    List[Tuple[int, str]]
        List of (index, smiles) tuples for molecules that pass the filtering
    """
    filtered_molecules = []
    
    for i, smi in enumerate(smiles_list):
        # Skip empty entries
        if not smi or not isinstance(smi, str):
            continue
            
        # Quick check with cached validation
        if not _is_valid_smiles(smi):
            continue
            
        try:
            mol = oechem.OEMol()
            if not oechem.OESmilesToMol(mol, smi):
                continue
                
            # Quick structure checks
            heavy_atoms = oechem.OECount(mol, oechem.OEIsHeavy())
            if heavy_atoms > max_heavy or heavy_atoms < 3:
                continue
                
            rotatable_bonds = oechem.OECount(mol, oechem.OEIsRotor())
            if rotatable_bonds > max_rot:
                continue
                
            # Optional complexity check for larger molecules
            if heavy_atoms > 30 or rotatable_bonds > 10:
                complexity = _calculate_molecule_complexity(smi)
                if complexity > max_complexity:
                    continue
                    
            # All checks passed, add to filtered list
            filtered_molecules.append((i, smi))
        except Exception:
            # Skip molecules that cause errors
            continue
            
    return filtered_molecules

def _calculate_optimal_workers_for_workload(total_molecules, use_gpu):
    """
    Calculate optimal number of worker processes based on workload size and system resources.
    
    Parameters
    ----------
    total_molecules : int
        Total number of molecules to process
    use_gpu : bool
        Whether GPU mode is active
        
    Returns
    -------
    int
        Optimal number of worker processes
    """
    if use_gpu:
        return 1  # Single process for GPU to avoid contention
        
    # Base calculation on number of molecules
    if total_molecules < 5000:
        # For small workloads, use fewer workers to reduce overhead
        worker_count = max(1, min(2, os.cpu_count() - 2)) if os.cpu_count() else 1
    elif total_molecules < 20000:
        # Medium workloads - balanced approach
        worker_count = max(2, min(4, os.cpu_count() // 2)) if os.cpu_count() else 2
    else:
        # Large workloads - maximize parallelism but leave room for system
        worker_count = max(4, os.cpu_count() - 4) if os.cpu_count() else 4
    
    # Further adjust based on memory
    if PSUTIL_AVAILABLE:
        total_memory, available_memory = _get_memory_info()
        # Estimate memory needed per worker
        est_memory_per_worker = _TARGET_MEMORY_PER_WORKER
        # Reduce workers if memory constrained
        memory_based_workers = int(available_memory / est_memory_per_worker)
        worker_count = min(worker_count, max(1, memory_based_workers))
    
    return worker_count

def _prepare_shared_molecule_database(smiles_list: List[str], idxs: List[int], 
                                   max_iso: int, max_rot: int, max_heavy: int, 
                                   use_gpu: bool, max_confs: int = 10) -> Tuple[str, Dict[str, int]]:
    """
    Prepare molecules for scoring by filtering, generating isomers and conformers.
    Creates a persistent database that can be shared across multiple models.
    
    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings to prepare
    idxs : List[int]
        Original indices for the SMILES strings
    max_iso : int
        Maximum number of isomers to generate
    max_rot : int
        Maximum number of rotatable bonds to consider
    max_heavy : int
        Maximum number of heavy atoms to process
    use_gpu : bool
        Whether to use GPU for conformer generation when available
    max_confs : int
        Maximum number of conformers to generate per molecule
        
    Returns
    -------
    Tuple[str, Dict[str, int]]
        Path to the prepared molecule database and title-to-index mapping dictionary
    """
    # Apply filtering first to reduce workload
    filtered_data = _prefilter_molecules(smiles_list, max_rot, max_heavy)
    
    if not filtered_data:
        return "", {}
    
    # Generate title to parent mapping and get conformers
    isomers = []
    title2parent = {}
    
    # Check if GPU is available for Omega
    omega_gpu = use_gpu
    if use_gpu:
        try:
            omega_gpu = oeomega.OEOmegaIsGPUReady()
            if not omega_gpu:
                print("Warning: GPU requested but Omega GPU is not ready, using CPU for conformer generation")
        except Exception as e:
            print(f"Warning: Error checking Omega GPU status, using CPU: {e}")
            omega_gpu = False
    
    # Use cached conformer generation options
    omega = oeomega.OEOmega()
    omega.SetOptions(_get_omega_options(omega_gpu, max_confs))
    omega.SetMaxConfs(max_confs)
    
    # Process molecules in batches to manage memory
    batch_size = 50  # Small batch size to avoid memory issues during preparation
    
    # Create shared temp directory for the database
    db_dir = os.path.join(_CACHE_DIR, "shared_databases")
    os.makedirs(db_dir, exist_ok=True)
    
    # Create a unique filename for this database
    import hashlib
    db_hash = hashlib.md5(str(filtered_data).encode()).hexdigest()[:10]
    database_path = os.path.join(db_dir, f"shared_db_{db_hash}.oeb")
    
    # Check if database already exists (from previous run)
    if os.path.exists(database_path):
        # Load existing database to extract title2parent mapping
        mdb = oechem.OEMolDatabase()
        if mdb.Open(database_path):
            print(f"Using existing shared database at {database_path}")
            # Extract title to parent mapping
            for mol_idx in range(mdb.GetMaxMolIdx()):
                mol = oechem.OEGraphMol()
                if mdb.GetMolecule(mol, mol_idx):
                    title = mol.GetTitle()
                    if "+" in title:  # It's a conformer with format "idx+isomer"
                        parent_id = int(title.split("+")[0])
                        title2parent[title] = parent_id
            return database_path, title2parent
    
    # Process molecules in batches
    with oechem.oemolostream() as ofs:
        if use_gpu:
            oechem.OEPRECompress(ofs)
            
        if not ofs.open(database_path):
            print(f"Error: Could not open output file {database_path}")
            return "", {}
            
        for i in range(0, len(filtered_data), batch_size):
            batch = filtered_data[i:i+batch_size]
            
            # Generate conformers for this batch
            for idx, smi in batch:
                try:
                    # Convert to OEMol and generate conformers
                    mol = oechem.OEMol()
                    if oechem.OESmilesToMol(mol, smi):
                        mol.SetTitle(str(idx))
                        
                        # Generate isomers
                        for iso in _enumerate_isomers(mol, max_iso=max_iso):
                            # Generate conformers
                            omega(iso)
                            for conf in iso.GetConfs():
                                confmol = oechem.OEMol(conf)
                                # Store title to parent mapping
                                title = confmol.GetTitle()
                                title2parent[title] = idx
                                
                                # Prepare for GPU if needed
                                if use_gpu:
                                    try:
                                        oefastrocs.OEPrepareFastROCSMol(confmol)
                                        half_mol = oechem.OEMol(confmol, oechem.OEMCMolType_HalfFloatCartesian)
                                        oechem.OEWriteMolecule(ofs, half_mol)
                                    except Exception as e:
                                        # Fall back to standard preparation
                                        oechem.OEWriteMolecule(ofs, confmol)
                                else:
                                    oechem.OEWriteMolecule(ofs, confmol)
                except Exception as e:
                    print(f"Error preparing molecule {idx}: {e}")
                    
            # Force garbage collection after each batch
            gc.collect()
    
    print(f"Created shared molecule database at {database_path} with {len(title2parent)} conformers")
    return database_path, title2parent

def _score_with_shared_database(database_path: str, title2parent: Dict[str, int], 
                               sq_model: str, use_gpu: bool) -> Dict[int, float]:
    """
    Score molecules using a shared database file.
    
    Parameters
    ----------
    database_path : str
        Path to the shared molecule database
    title2parent : Dict[str, int]
        Mapping from molecule titles to original indices
    sq_model : str
        Path to the shape query file
    use_gpu : bool
        Whether to use GPU acceleration
        
    Returns
    -------
    Dict[int, float]
        Dictionary mapping original indices to scores
    """
    if not database_path or not title2parent:
        return {}
        
    # Get or create shape database, query, and options
    try:
        db, query, opts = _SHAPE_DB_CACHE.get_or_create_database(sq_model, use_gpu)
    except Exception as e:
        print(f"Error creating shape database: {e}")
        return {}
    
    # Prepare the molecules from the shared database
    scores: Dict[int, float] = {}
    
    try:
        # Open the shared molecule database
        mdb = oechem.OEMolDatabase()
        if not mdb.Open(database_path):
            print(f"Error: Could not open shared database {database_path}")
            return {}
        
        # Create a fresh database for scoring
        fresh_db = oefastrocs.OEShapeDatabase()
        fresh_db.SetNumOpenThreads(1)  # Conservative thread count
        
        # Open shape database with molecule database
        if not fresh_db.Open(mdb):
            print("Error: Could not open shape database with molecule database")
            return {}
        
        # Process scores
        for sc in fresh_db.GetSortedScores(query, opts):
            mol_idx = sc.GetMolIdx()
            mol_title = mdb.GetTitle(mol_idx)
            parent = title2parent.get(mol_title)
            
            if parent is not None:
                tc = sc.GetTanimotoCombo()
                scores[parent] = max(tc, scores.get(parent, 0.0))
    except Exception as e:
        print(f"Error during molecule scoring: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        fresh_db = None
        mdb = None
        gc.collect()
        
    return scores
