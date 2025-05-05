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

Key points
----------
✓  GPU : optimized single‑process with efficient memory usage
✓  CPU : adaptive worker count with optimized resource usage
✓  Persistent database caching for repeated calculations
✓  Memory-optimized workflow with early filtering
✓  Adaptive batch sizing based on system resources
✓  Interface compatible with the original code (`getScores`, `__call__`)
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
        
        # Calculate workers based on memory constraints
        memory_workers = max(1, int(usable_memory / _TARGET_MEMORY_PER_WORKER))
        
        # Calculate workers based on CPU - leave at least 1 core free
        cpu_workers = max(1, cpu_count - 1)
        
        # Use the minimum of memory-based and CPU-based calculations
        optimal = min(memory_workers, cpu_workers)
        
        # Honor user suggestion if provided, but don't exceed system capacity
        if suggested_workers is not None:
            return min(suggested_workers, optimal)
        return optimal
    else:
        # Fallback without psutil
        cpu_count = os.cpu_count() or 4
        return suggested_workers if suggested_workers is not None else max(1, cpu_count - 1)

# Global flag to track memory pool initialization
_OE_MEMORY_POOL_INITIALIZED = False

def _init_worker(worker_id=None):
    """Enhanced initializer for fork‑server workers."""
    global _OE_MEMORY_POOL_INITIALIZED
    
    # Configure OpenEye - only initialize memory pool if not done already
    if not _OE_MEMORY_POOL_INITIALIZED and 'OE_MEMORY_POOL_SET' not in os.environ:
        oechem.OESetMemPoolMode(oechem.OEMemPoolMode_System)
        _OE_MEMORY_POOL_INITIALIZED = True
        os.environ['OE_MEMORY_POOL_SET'] = '1'
    
    os.environ["OE_SILENT"] = "true"
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
            for root, _, files in os.walk(path, topdown=False):
                for f in files:
                    try: os.remove(os.path.join(root, f))
                    except Exception: pass
            try: os.rmdir(path)
            except Exception: pass


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

def filter_molecules(smiles_list: List[str], max_rot: int = 10, max_heavy: int = 30) -> List[Tuple[str, bool]]:
    """
    Filter molecules for FastROCS processing, enhanced for performance.
    Returns list of (smiles, is_valid) tuples. Uses parallel processing for large lists.
    """
    # For small lists, process directly
    if len(smiles_list) <= 100:
        return _filter_molecules_chunk(smiles_list, max_rot, max_heavy)
        
    # For larger lists, use parallel processing with thread pool
    chunk_size = 50
    chunks = [smiles_list[i:i+chunk_size] for i in range(0, len(smiles_list), chunk_size)]
    
    results = []
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as executor:
        chunk_results = list(executor.map(
            lambda chunk: _filter_molecules_chunk(chunk, max_rot, max_heavy), chunks
        ))
        
    # Flatten results
    for cr in chunk_results:
        results.extend(cr)
            
    return results

def _filter_molecules_chunk(smiles_list: List[str], max_rot: int, max_heavy: int) -> List[Tuple[str, bool]]:
    """Process a chunk of molecules for filtering. Helper for parallel processing."""
    results = []
    compiled_patterns = _get_compiled_patterns()
    
    for smi in smiles_list:
        # Quick check with cached validation
        if not _is_valid_smiles(smi):
            results.append((smi, False))
            continue
            
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smi)  # We already validated above
        
        # Quick check for problematic atoms (most common rejection reason)
        if any(oechem.OEGetAtomicSymbol(a.GetAtomicNum()) in _BAD_ATOMS for a in mol.GetAtoms()):
            results.append((smi, False))
            continue
            
        # Check rotatable bonds and heavy atoms
        rotatable_bonds = oechem.OECount(mol, oechem.OEIsRotor())
        heavy_atoms = oechem.OECount(mol, oechem.OEIsHeavy())
        
        if rotatable_bonds > max_rot or heavy_atoms > max_heavy:
            results.append((smi, False))
            continue
            
        # Check against all patterns
        atom_count = mol.NumAtoms()
        if atom_count > 100 or atom_count < 3:
            results.append((smi, False))
            continue
            
        if any(pat.SingleMatch(mol) for pat in compiled_patterns):
            results.append((smi, False))
            continue
        
        # Check connectivity - a molecule should be a single connected component
        visited = [False] * mol.NumAtoms()
        components = 0
        
        def dfs(atom_idx):
            visited[atom_idx] = True
            for bond in mol.GetAtom(oechem.OEHasAtomIdx(atom_idx)).GetBonds():
                next_atom_idx = bond.GetNbr(mol.GetAtom(oechem.OEHasAtomIdx(atom_idx))).GetIdx()
                if not visited[next_atom_idx]:
                    dfs(next_atom_idx)
        
        for i in range(mol.NumAtoms()):
            if not visited[i]:
                components += 1
                dfs(i)
                if components > 1:
                    break
        
        if components > 1:
            results.append((smi, False))  # Multiple components
        else:
            results.append((smi, True))   # Valid molecule
            
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
    omegaOpts.GetTorDriveOptions().SetUseGPU(use_gpu)
    try:
        omegaOpts.GetTorDriveOptions().SetForceField(oeff.OEMMFFSheffieldFFType_MMFF94s)
    except:
        pass
    
    omegaOpts.SetStrictStereo(False)
    omegaOpts.SetFromCT(True)
    
    builder_opts = omegaOpts.GetMolBuilderOptions()
    builder_opts.SetSampleHydrogens(False)
    
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
            
            # Create new database and query
            db = oefastrocs.OEShapeDatabase()
            
            # Set appropriate thread count based on mode
            if use_gpu:
                # GPU mode uses a single thread
                db.SetNumOpenThreads(1)
            else:
                # For CPU mode, use multiple threads
                db.SetNumOpenThreads(max(1, min(4, os.cpu_count() or 2)))
            
            # Create options with correct mode setting
            opts = oefastrocs.OEShapeDatabaseOptions()
            if use_gpu:
                # Use GPU mode if available
                opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_FastROCS)
            else:
                # Otherwise use ROCS mode for CPU
                opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_ROCS)
            
            # Create query
            query = oeshape.OEShapeQuery()
            if not oeshape.OEReadShapeQuery(sq_model_path, query):
                raise ValueError(f"Invalid shape query file: {sq_model_path}")
            
            self.databases[key] = (db, query, opts)
            return db, query, opts

# Global database cache
_SHAPE_DB_CACHE = ShapeDatabaseCache()

def _prepare_molecules_for_scoring(smiles_list: List[str], idxs: List[int], 
                                  max_iso: int, max_rot: int, max_heavy: int, 
                                  use_gpu: bool) -> Tuple[List[oechem.OEMol], Dict[str, int]]:
    """
    Prepare molecules for scoring by filtering, generating isomers and conformers.
    Returns list of conformers and title-to-index mapping.
    """
    # Apply unified filtering to all molecules in batch
    filtered_data = []
    filter_results = filter_molecules(smiles_list, max_rot, max_heavy)
    
    for (smi, is_valid), idx in zip(filter_results, idxs):
        if is_valid:
            filtered_data.append((smi, idx))
    
    if not filtered_data:
        return [], {}
    
    title2parent: Dict[str, int] = {}
    isomers: List[oechem.OEMol] = []

    # Use cached conformer generation options
    omega = oeomega.OEOmega()
    omega.SetOptions(_get_omega_options(use_gpu))
    omega.SetMaxConfs(10)
    
    # Use threading for conformer generation when processing many molecules
    if len(filtered_data) > 10 and not use_gpu:
        # For larger batches in CPU mode, use thread parallelism for conformer generation
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as executor:
            futures = []
            for s, idx in filtered_data:
                futures.append(executor.submit(_generate_conformers, s, str(idx), omega, max_iso))
            
            for future in futures:
                result = future.result()
                if result:
                    mol_title2parent, mol_isomers = result
                    title2parent.update(mol_title2parent)
                    isomers.extend(mol_isomers)
    else:
        # For smaller batches or GPU mode, process sequentially
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
        # We don't need to check validity - OEReadShapeQuery already does that during creation
    except Exception as e:
        print(f"Error creating shape database: {e}")
        return {}
    
    # Prepare the molecules in a temporary database
    scores: Dict[int, float] = {}
    
    # Use cache directory for database if it's CPU mode (more reusable)
    with _tmpdir(prefix="rocs_mols", use_cache=not use_gpu, 
                 cache_key=_get_file_hash(sq_model) if not use_gpu else None) as td:
        sdf = os.path.join(td, "confs.sdf")
        with oechem.oemolostream(sdf) as ofs:
            for m in isomers:
                oechem.OEWriteMolecule(ofs, m)

        # Create molecule database
        mdb = oechem.OEMolDatabase()
        if not mdb.Open(sdf):
            return {}
        
        # Create a fresh database for each batch to avoid the "already contains data" error
        fresh_db = oefastrocs.OEShapeDatabase()
        # Copy settings from cached database
        fresh_db.SetNumOpenThreads(db.GetNumOpenThreads())
        
        # Open shape database with molecule database
        if not fresh_db.Open(mdb):
            return {}
        
        # Process scores in batches for memory efficiency
        # Use the fresh database with the cached query and options
        for sc in fresh_db.GetSortedScores(query, opts):
            mol_idx = sc.GetMolIdx()
            dbmol = oechem.OEMol()
            if mdb.GetMolecule(dbmol, mol_idx):
                parent = title2parent.get(dbmol.GetTitle())
                if parent is not None:
                    tc = sc.GetTanimotoCombo()
                    scores[parent] = max(tc, scores.get(parent, 0.0))
        
    return scores

def _score_batch(batch: Tuple[List[str], List[int]],
                 sq_model: str,
                 max_iso: int,
                 max_rot: int,
                 max_heavy: int,
                 use_gpu: bool = True,
                 worker_id: int = None) -> Dict[int, float]:
    """
    Process and score a batch of molecules.
    Combines filtering, conformer generation, and scoring in one efficient function.
    Optimized with caching and improved memory management.
    """
    start_time = time.time()
    smiles, idxs = batch
    
    # Set CPU affinity if this is a worker process
    if worker_id is not None and not use_gpu:
        os.environ['WORKER_ID'] = str(worker_id)
        _init_worker(worker_id)
    
    # Prepare molecules (filter, generate conformers)
    isomers, title2parent = _prepare_molecules_for_scoring(
        smiles, idxs, max_iso, max_rot, max_heavy, use_gpu
    )
    
    if not isomers:
        return {}
    
    # Score molecules
    scores = _score_molecules_with_database(isomers, title2parent, sq_model, use_gpu)
    
    # Clean up to reduce memory usage
    isomers.clear()
    gc.collect()
    
    return scores


# ------------------------------------------------------------------------------
#  Scorer class
# ------------------------------------------------------------------------------

class OpenEyeScorer(Scorer):
    """
    Compatible with original code: expose getScores(list[str]) and __call__(…).
    """

    def __init__(self,
                 sq_model_path: str,
                 use_gpu: bool = True,
                 max_isomers: int = 4,
                 max_rot_bonds: int = 10,
                 max_heavy_atoms: int = 30,
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
        cpu_processes : int | None, optional
            Number of CPU processes to use if not using GPU. If None, will use
            available CPU cores minus 2 (to leave resources for the system).
            Ignored when GPU mode is active.
        """
        if not os.path.isfile(sq_model_path):
            raise FileNotFoundError(sq_model_path)

        self.sq_model = sq_model_path
        self.max_iso  = max_isomers
        self.max_rot  = max_rot_bonds
        self.max_heavy = max_heavy_atoms

        # ------------------------------------------------------------------
        #   Device selection
        # ------------------------------------------------------------------
        gpu_ready = use_gpu and oefastrocs.OEFastROCSIsGPUReady()
        self.use_gpu = gpu_ready
        if gpu_ready:
            self.cpu_procs = 0
            print("FastROCS GPU mode   : ON  (single process)")
        else:
            avail = max(1, mp.cpu_count() - 2)
            self.cpu_procs = cpu_processes if cpu_processes else avail
            print(f"FastROCS CPU mode   : {self.cpu_procs} fork‑server workers")

        os.environ["OE_SILENT"] = "true"

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
        Optimized with adaptive batching and better resource utilization.
        """
        if not smiles:
            return np.zeros(0)
            
        # Calculate optimal batch size based on available memory and system resources
        if PSUTIL_AVAILABLE:
            _, available_memory = _get_memory_info()
            # Adjust batch size based on available memory (smaller when memory is tight)
            base_batch_size = max(_MIN_BATCH_SIZE, min(_MAX_BATCH_SIZE, 
                                                     int(available_memory * 10)))
            
            # Further adjust based on molecule complexity - sample a few to estimate
            sample_size = min(50, len(smiles))
            sample_smiles = smiles[:sample_size] if sample_size > 0 else smiles
            complex_mol_ratio = 0.0
            
            filter_results = filter_molecules(sample_smiles, self.max_rot, self.max_heavy)
            valid_count = sum(1 for _, valid in filter_results if valid)
            if sample_size > 0:
                complex_mol_ratio = 1.0 - (valid_count / sample_size)
            
            # Reduce batch size for complex molecules that have high memory requirements
            complexity_factor = 1.0 + (complex_mol_ratio * 2.0)  # Scale from 1.0 to 3.0
            adjusted_batch_size = max(_MIN_BATCH_SIZE, int(base_batch_size / complexity_factor))
        else:
            # Default to conservative batch size if we can't measure system resources
            adjusted_batch_size = 30
            
        # Prepare batches with the optimized size
        idxs = list(range(len(smiles)))
        batches = []
        for i in range(0, len(smiles), adjusted_batch_size):
            end_idx = min(i + adjusted_batch_size, len(smiles))
            batches.append((smiles[i:end_idx], idxs[i:end_idx]))
            
        # For GPU mode: use single-process scoring with optimized memory handling
        if self.use_gpu:
            results = {}
            start_time = time.time()
            
            # Process each batch and measure timing for adaptive optimization
            for i, batch in enumerate(batches):
                batch_start = time.time()
                batch_results = _score_batch(
                    batch, self.sq_model, self.max_iso, 
                    self.max_rot, self.max_heavy, True
                )
                results.update(batch_results)
                
                # Force garbage collection between large batches
                if len(batch[0]) > 50:
                    gc.collect()
                    
                # Log progress for long-running jobs
                batch_time = time.time() - batch_start
                if i % 5 == 0 and i > 0:
                    print(f"GPU processed {i}/{len(batches)} batches, " 
                          f"avg time: {(time.time() - start_time) / i:.2f}s per batch")
                
        # For CPU mode: use optimized multiprocessing with better resource management
        else:
            # Calculate optimal worker count based on current system state
            optimal_workers = _calculate_optimal_workers(self.cpu_procs)
            ctx = mp.get_context("forkserver")
            results = {}
            
            with ProcessPoolExecutor(max_workers=optimal_workers,
                                    mp_context=ctx) as pool:
                # Pass worker ID to each process for better CPU affinity
                futures = []
                for i, batch in enumerate(batches):
                    worker_id = i % optimal_workers
                    futures.append(pool.submit(
                        _score_batch,
                        batch=batch,
                        sq_model=self.sq_model,
                        max_iso=self.max_iso,
                        max_rot=self.max_rot,
                        max_heavy=self.max_heavy,
                        use_gpu=False,
                        worker_id=worker_id
                    ))
                
                # Process results as they complete
                for i, future in enumerate(futures):
                    try:
                        batch_results = future.result()
                        results.update(batch_results)
                        
                        # Log progress for long-running jobs
                        if i % 10 == 0 and i > 0:
                            print(f"CPU processed {i}/{len(futures)} batches")
                    except Exception as e:
                        print(f"Error in worker process: {e}")
        
        # Convert dictionary to array
        out = np.zeros(len(smiles))
        for k, v in results.items():
            out[k] = v
            
        # Final cleanup to minimize memory usage after processing
        gc.collect()
            
        return out

    # ------------------------------------------------------------------
    def getKey(self):
        return "ROCS"
