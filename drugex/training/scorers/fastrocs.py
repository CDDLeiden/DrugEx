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
            oechem.OESetMemPoolMode(oechem.OEMemPoolMode_System)
            _OE_MEMORY_POOL_INITIALIZED = True
            os.environ["OE_MEMORY_POOL_INITIALIZED"] = "true"
            print("OpenEye memory pool initialized in fastrocs module")
        except Exception as e:
            print(f"Warning: Failed to set memory pool mode: {e}")

# Global flag to track memory pool initialization
_OE_MEMORY_POOL_INITIALIZED = False
# Initialize at module import time
_initialize_oe_memory_pool()

# Create persistent cache directories
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "fastrocs_cache")
_DB_CACHE_DIR = os.path.join(_CACHE_DIR, "databases")
_CONF_CACHE_DIR = os.path.join(_CACHE_DIR, "conformers")

# Create cache directories if they don't exist
for d in [_CACHE_DIR, _DB_CACHE_DIR, _CONF_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# Thread-safe cache for database handles
_DB_CACHE = {}
_DB_CACHE_LOCK = threading.RLock()

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

# Extended list of problematic atoms for better filtering
_BAD_ATOMS = {'Au','Ag','Al','As','Be','Bi','Ce','Dy','Eu','Gd','Hf','Hg','Ho','In','Ir','La',
             'Lu','Nd','Os','Pd','Pm','Pr','Pt','Re','Rh','Ru','Sm','Ta','Tb','Th','Ti','Tm',
             'U','V','W','Y','Yb','Zr'}

# Cache for SMILES validation, significantly speeds up repeated checks
@lru_cache(maxsize=1000)
def _is_valid_smiles(smiles: str) -> bool:
    """Cached SMILES validation check."""
    if not smiles or not isinstance(smiles, str) or len(smiles) < 2:
        return False
        
    mol = oechem.OEMol()
    return bool(oechem.OESmilesToMol(mol, smiles))

# Cache for compiled SMARTS patterns
_PATTERN_CACHE = {}

def _get_compiled_patterns():
    """Get cached compiled SMARTS patterns for molecular filtering."""
    if not _PATTERN_CACHE:
        # Define SMARTS patterns for problematic structures
        problem_patterns = [
            '[S+]', '[n+]', '[N+](=[O-])', '[#7,#16]~[#7,#16]',
            '[C,c]#[C,c]', '[#6]=[#6]=[#6]', '[r3]', '[Si]', '[P]'
        ]
        
        # Compile patterns once
        for pattern in problem_patterns:
            pat = oechem.OESubSearch()
            if pat.Init(pattern):
                _PATTERN_CACHE[pattern] = pat
    
    return list(_PATTERN_CACHE.values())

def filter_molecules(smiles_list: List[str], max_rot: int = 15, max_heavy: int = 45) -> List[Tuple[str, bool]]:
    """
    Filter molecules for FastROCS processing with relaxed criteria to match ez_rocs.
    Returns list of (smiles, is_valid) tuples.
    """
    # For small lists, process directly
    if len(smiles_list) <= 100:
        return _filter_molecules_chunk(smiles_list, max_rot, max_heavy)
        
    # For larger lists, use parallel processing with thread pool
    chunk_size = 50
    chunks = [smiles_list[i:i+chunk_size] for i in range(0, len(smiles_list), chunk_size)]
    
    results = []
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as executor:
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
    Matches ez_rocs behavior to accept more molecules.
    """
    results = []
    
    for smi in smiles_list:
        # Quick check with cached validation
        if not _is_valid_smiles(smi):
            results.append((smi, False))
            continue
            
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smi)
        
        # Relaxed filtering - only check the most essential criteria
        rotatable_bonds = oechem.OECount(mol, oechem.OEIsRotor())
        heavy_atoms = oechem.OECount(mol, oechem.OEIsHeavy())
        
        if rotatable_bonds > max_rot or heavy_atoms > max_heavy:
            results.append((smi, False))
            continue
            
        # Basic structural check - molecule should have at least a few atoms
        atom_count = mol.NumAtoms()
        if atom_count < 3:
            results.append((smi, False))
            continue
        
        # Accept more molecules - match ez_rocs behavior
        results.append((smi, True))
            
    return results

# Cached flipper options to avoid recreating them for each molecule
@lru_cache(maxsize=10)
def _get_flipper_options(max_centers=4):
    """Cached flipper options for isomer enumeration."""
    opts = oeomega.OEFlipperOptions()
    opts.SetMaxCenters(max_centers)
    return opts

def _enumerate_isomers(mol: oechem.OEMol, max_centers=4, max_iso=4):
    """Generate isomers for a molecule, optimized with cached options."""
    opts = _get_flipper_options(max_centers)
    for i, conf in enumerate(oeomega.OEFlipper(mol, opts)):
        if i == max_iso:
            break
        iso = oechem.OEMol(conf)
        iso.SetTitle(f"{mol.GetTitle()}+{i}")
        yield iso

# Cached omega options for conformer generation
@lru_cache(maxsize=10)
def _get_omega_options(use_gpu: bool, max_confs: int = 10):
    """Get cached omega options for conformer generation."""
    omegaOpts = oeomega.OEOmegaOptions()
    
    # Configure GPU mode for TorDrive
    try:
        if use_gpu and oeomega.OEOmegaIsGPUReady():
            # Enable GPU mode
            omegaOpts.GetTorDriveOptions().SetUseGPU(True)
            # For GPU compatibility, use recommended force field
            from openeye import oeff
            omegaOpts.GetTorDriveOptions().SetForceField(oeff.OEMMFFSheffieldFFType_MMFF94Smod_NOESTAT)
            # Disable hydrogen sampling for GPU compatibility
            omegaOpts.GetMolBuilderOptions().SetSampleHydrogens(False)
            print("Omega GPU mode enabled for conformer generation")
        else:
            omegaOpts.GetTorDriveOptions().SetUseGPU(False)
    except Exception as e:
        print(f"Warning: Error configuring Omega GPU mode: {e}")
        omegaOpts.GetTorDriveOptions().SetUseGPU(False)
    
    # Common settings for both CPU and GPU modes
    omegaOpts.SetStrictStereo(False)
    omegaOpts.SetFromCT(True)
    omegaOpts.SetMaxConfs(max_confs)
    
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
            if not oeshape.OEReadShapeQuery(sq_model_path, query):
                raise ValueError(f"Invalid shape query file: {sq_model_path}")
            
            # Create options with correct mode setting
            opts = oefastrocs.OEShapeDatabaseOptions()
            if use_gpu:
                # Use GPU mode if available and explicitly set FastROCS mode
                opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_FastROCS)
                print("FastROCS GPU mode enabled for shape queries")
            else:
                # CPU mode (ROCS) - explicitly set
                opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_ROCS)
            
            # Create new database
            db = oefastrocs.OEShapeDatabase()
            
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
    # Apply simplified filtering to match ez_rocs behavior
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
    oechem.OESmilesToMol(mol, smiles)
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
    except oechem.OELicenseError as e:
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
        fresh_db = oefastrocs.OEShapeDatabase()
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
        except oechem.OELicenseError as e:
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
#  Scorer class
# ------------------------------------------------------------------------------

class OpenEyeScorer(Scorer):
    """
    A FastROCS-based scorer with optimized performance for both GPU and CPU modes.
    Compatible with the original DrugEx scorer interface.
    """

    def __init__(self,
                 sq_model_path: str,
                 use_gpu: bool = True,
                 max_isomers: int = 4,
                 max_rot_bonds: int = 10,
                 max_heavy_atoms: int = 30,
                 max_conformers: int = 10,
                 cpu_processes: int | None = None):
        """
        Initialize the OpenEye FastROCS scorer.

        Parameters
        ----------
        sq_model_path : str
            Path to the ROCS query file (.sq file)
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
        """
        if not os.path.isfile(sq_model_path):
            raise FileNotFoundError(sq_model_path)

        # Memory pool initialization is now handled at the module level
        # to prevent duplicate calls across different parts of the program

        self.sq_model = sq_model_path
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
            # Use more conservative process count for CPU mode
            if cpu_processes is not None:
                self.cpu_procs = min(2, cpu_processes)
            else:
                self.cpu_procs = min(2, _calculate_optimal_workers())
            print(f"FastROCS CPU mode   : {self.cpu_procs} fork‑server workers")

        # Validate the query file and prepare
        try:
            query = oeshape.OEShapeQuery()
            if not oeshape.OEReadShapeQuery(sq_model_path, query):
                raise ValueError(f"Invalid shape query file: {sq_model_path}")
                
            # Create database options with correct mode
            opts = oefastrocs.OEShapeDatabaseOptions()
            if self.use_gpu:
                opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_FastROCS)
            else:
                opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_ROCS)
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
                smiles.append(Chem.MolToSmiles(m))
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
                        batch, self.sq_model, self.max_iso, 
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
                            batch, self.sq_model, self.max_iso, 
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
                            sq_model=self.sq_model,
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
    with oechem.oemolostream() as ofs:
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
                    half_mol = oechem.OEMol(mol, oechem.OEMCMolType_HalfFloatCartesian)
                    oechem.OEWriteMolecule(ofs, half_mol)
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
    mdb = oechem.OEMolDatabase()
    if not mdb.Open(output_path):
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
