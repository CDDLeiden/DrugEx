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
- Optimized implementation for maximum performance
- Memory management techniques for large-scale processing
- Multi-threading and process pool support for CPU parallelization
- GPU acceleration with optimized data handling
- Resource-aware batch sizing and efficient caching
- Suitable for high-throughput screening
- Designed to handle thousands of molecules reliably

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

# ===============================================================================
# CRITICAL SEGMENTATION FAULT FIXES - Enhanced Memory Management
# ===============================================================================

# Global configuration and caching - Enhanced
_OE_MEMORY_POOL_INITIALIZED = False
_WORKER_INITIALIZED = False

# Enhanced cache management with cleanup
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "fastrocs_cache")
_DB_CACHE_DIR = os.path.join(_CACHE_DIR, "databases")
_CONF_CACHE_DIR = os.path.join(_CACHE_DIR, "conformers")

# More conservative batch sizing to prevent memory issues
_MIN_BATCH_SIZE = 5       # Reduced from 10
_MAX_BATCH_SIZE = 50      # Reduced from 200  
_TARGET_MEMORY_PER_WORKER = 1.0  # Reduced from 1.5 GB

# Limits to prevent molecule explosion while allowing base_rocs.py alignment
_MAX_TOTAL_CONFORMERS = 2000  # Increased to handle 200 conformers per molecule
_MAX_MOLECULES_PER_DB = 100  # Limit molecules per database

# Create cache directories with error handling
for d in [_CACHE_DIR, _DB_CACHE_DIR, _CONF_CACHE_DIR]:
    try:
        os.makedirs(d, exist_ok=True)
    except (OSError, IOError) as e:
        print(f"Warning: Could not create cache directory {d}: {e}")

def _initialize_oe_memory_pool():
    """Enhanced OpenEye memory pool initialization with better error handling."""
    global _OE_MEMORY_POOL_INITIALIZED
    
    # Check if already initialized in this process
    if os.environ.get("OE_MEMORY_POOL_INITIALIZED") == "true":
        _OE_MEMORY_POOL_INITIALIZED = True
        return True
        
    if not _OE_MEMORY_POOL_INITIALIZED:
        try:
            # Set memory pool mode with better error handling
            oechem.OESetMemPoolMode(oechem.OEMemPoolMode_System)
            _OE_MEMORY_POOL_INITIALIZED = True
            os.environ["OE_MEMORY_POOL_INITIALIZED"] = "true"
            return True
        except Exception as e:
            print(f"Warning: Failed to set OpenEye memory pool mode: {e}")
            return False
    return True

# Initialize at module import time
_initialize_oe_memory_pool()

# ===============================================================================
# Enhanced Generic helpers with better resource management
# ===============================================================================

def _get_memory_info():
    """Get system memory information with fallback."""
    if PSUTIL_AVAILABLE:
        try:
            vm = psutil.virtual_memory()
            return vm.total / (1024**3), vm.available / (1024**3)
        except Exception:
            pass
    # Conservative fallback
    return 4.0, 2.0

def _calculate_optimal_workers(suggested_workers=None):
    """Calculate optimal workers with more conservative estimates."""
    if PSUTIL_AVAILABLE:
        try:
            total_memory, available_memory = _get_memory_info()
            cpu_count = os.cpu_count() or 2
            
            # More conservative memory allocation
            reserved_memory = max(3.0, total_memory * 0.35)  # Increased reservation
            usable_memory = max(0.5, available_memory - reserved_memory)
            
            # More conservative worker calculation
            memory_workers = max(1, int(usable_memory / (_TARGET_MEMORY_PER_WORKER * 2.0)))
            cpu_workers = max(1, min(2, max(1, cpu_count - 3)))  # Leave more cores free
            
            optimal = min(memory_workers, cpu_workers, 2)  # Cap at 2 workers max
            
            if suggested_workers is not None:
                return min(suggested_workers, optimal)
            return optimal
        except Exception:
            pass
    
    # Ultra-conservative fallback
    return 1 if suggested_workers is None else min(suggested_workers, 1)

@contextmanager
def _tmpdir(prefix="fastrocs_", use_cache=False, cache_key=None):
    """Enhanced temporary directory manager with better cleanup."""
    if use_cache and cache_key:
        # Use cached directory but ensure it's clean
        path = os.path.join(_CACHE_DIR, f"{prefix}_{cache_key}")
        try:
            os.makedirs(path, exist_ok=True)
            # Clean any existing files in cached directory
            for existing_file in os.listdir(path):
                try:
                    file_path = os.path.join(path, existing_file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except (OSError, IOError):
                    pass
            yield path
        except Exception as e:
            print(f"Warning: Cache directory error: {e}")
            # Fallback to temporary directory
            path = tempfile.mkdtemp(prefix=prefix)
            try:
                yield path
            finally:
                _cleanup_directory(path)
    else:
        # Standard temporary directory with enhanced cleanup
        path = tempfile.mkdtemp(prefix=prefix)
        try:
            yield path
        finally:
            _cleanup_directory(path)

def _cleanup_directory(path):
    """Enhanced directory cleanup with better error handling."""
    if not os.path.exists(path):
        return
        
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            # Clean up files first
            for f in files:
                try:
                    file_path = os.path.join(root, f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except (OSError, IOError):
                    pass
            
            # Clean up directories
            for d in dirs:
                try:
                    dir_path = os.path.join(root, d)
                    if os.path.isdir(dir_path):
                        os.rmdir(dir_path)
                except (OSError, IOError):
                    pass
        
        # Finally remove the root directory
        try:
            os.rmdir(path)
        except (OSError, IOError):
            pass
    except Exception:
        pass  # Silent cleanup failure

# ===============================================================================
# Enhanced molecule utilities with better limits
# ===============================================================================

def filter_molecules(smiles_list: List[str], max_rot: int = 10, max_heavy: int = 35) -> List[Tuple[str, bool]]:
    """Enhanced molecule filtering with more conservative limits."""
    # More conservative limits by default
    if len(smiles_list) <= 50:
        return _filter_molecules_chunk(smiles_list, max_rot, max_heavy)
        
    # For larger lists, use thread pool with conservative settings
    chunk_size = 25  # Smaller chunks
    chunks = [smiles_list[i:i+chunk_size] for i in range(0, len(smiles_list), chunk_size)]
    
    results = []
    # Limit thread count for safety
    max_threads = min(2, os.cpu_count() or 1)
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        chunk_results = list(executor.map(
            lambda chunk: _filter_molecules_chunk(chunk, max_rot, max_heavy), chunks
        ))
        
    for cr in chunk_results:
        results.extend(cr)
            
    return results

def _filter_molecules_chunk(smiles_list: List[str], max_rot: int, max_heavy: int) -> List[Tuple[str, bool]]:
    """Enhanced filtering with stricter validation."""
    results = []
    
    for smi in smiles_list:
        try:
            # Quick cached validation
            if not _is_valid_smiles(smi):
                results.append((smi, False))
                continue
                
            mol = oechem.OEMol()
            if not oechem.OESmilesToMol(mol, smi):
                results.append((smi, False))
                continue
            
            # Enhanced filtering with stricter limits
            rotatable_bonds = oechem.OECount(mol, oechem.OEIsRotor())
            heavy_atoms = oechem.OECount(mol, oechem.OEIsHeavy())
            atom_count = mol.NumAtoms()
            
            # More conservative filtering
            if (rotatable_bonds > max_rot or 
                heavy_atoms > max_heavy or 
                atom_count < 3 or 
                atom_count > max_heavy + 20):  # Additional atom limit
                results.append((smi, False))
                continue
            
            results.append((smi, True))
            
        except Exception:
            results.append((smi, False))
            
    return results

# ===============================================================================
# Enhanced conformer generation with limits
# ===============================================================================

def _prepare_molecules_for_scoring(smiles_list: List[str], idxs: List[int], 
                                max_iso: int, max_rot: int, max_heavy: int, 
                                use_gpu: bool, max_confs: int = 10) -> Tuple[List[oechem.OEMol], Dict[str, int]]:
    """Enhanced molecule preparation with strict limits to prevent explosion."""
    
    # Apply enhanced filtering with more conservative limits
    filtered_data = []
    filter_results = filter_molecules(smiles_list, max_rot, max_heavy)
    
    for (smi, is_valid), idx in zip(filter_results, idxs):
        if is_valid:
            filtered_data.append((smi, idx))
    
    if not filtered_data:
        return [], {}
    
    # Limit total molecules to prevent memory explosion
    if len(filtered_data) > _MAX_MOLECULES_PER_DB:
        print(f"Warning: Limiting molecule count from {len(filtered_data)} to {_MAX_MOLECULES_PER_DB}")
        filtered_data = filtered_data[:_MAX_MOLECULES_PER_DB]
    
    title2parent: Dict[str, int] = {}
    isomers: List[oechem.OEMol] = []

    # Conservative conformer generation
    omega_gpu = use_gpu and _check_omega_gpu_safe()
    
    # Reduce parameters to prevent explosion while aligning with base_rocs.py
    effective_max_iso = min(max_iso, 4)  # Align with base_rocs.py default
    effective_max_confs = min(max_confs, 200)  # Align with base_rocs.py default
    
    omega = oeomega.OEOmega()
    omega.SetOptions(_get_omega_options(omega_gpu, effective_max_confs))
    omega.SetMaxConfs(effective_max_confs)
    
    total_conformers = 0
    for s, idx in filtered_data:
        # Check conformer limit before processing
        estimated_conformers = effective_max_iso * effective_max_confs
        if total_conformers + estimated_conformers > _MAX_TOTAL_CONFORMERS:
            print(f"Warning: Stopping molecule processing to prevent memory overflow")
            break
            
        result = _generate_conformers_safe(s, str(idx), omega, effective_max_iso)
        if result:
            mol_title2parent, mol_isomers = result
            title2parent.update(mol_title2parent)
            isomers.extend(mol_isomers)
            total_conformers += len(mol_isomers)

    print(f"Generated {len(isomers)} conformers from {len(filtered_data)} molecules")
    return isomers, title2parent

def _check_omega_gpu_safe() -> bool:
    """Safely check Omega GPU availability."""
    try:
        return oeomega.OEOmegaIsGPUReady()
    except Exception:
        return False

def _generate_conformers_safe(smiles: str, idx: str, omega: oeomega.OEOmega, max_iso: int) -> Tuple[Dict[str, int], List[oechem.OEMol]]:
    """Enhanced conformer generation with error handling."""
    try:
        mol = oechem.OEMol()
        if not oechem.OESmilesToMol(mol, smiles):
            return {}, []
            
        mol.SetTitle(idx)
        
        title2parent = {}
        isomers = []
        
        conformer_count = 0
        for iso in _enumerate_isomers(mol, max_iso=max_iso):
            if omega(iso):
                for conf in iso.GetConfs():
                    confmol = oechem.OEMol(conf)
                    title2parent[confmol.GetTitle()] = int(idx)
                    isomers.append(confmol)
                    conformer_count += 1
                    
                    # Safety limit aligned with base_rocs.py expectations
                    if conformer_count > 200:  # Match base_rocs.py max_conformers
                        break
                if conformer_count > 200:
                    break
                    
        return title2parent, isomers
        
    except Exception as e:
        print(f"Warning: Failed to generate conformers for molecule {idx}: {e}")
        return {}, []

# ===============================================================================
# Enhanced database management with better caching
# ===============================================================================

class EnhancedShapeDatabaseCache:
    """Enhanced database cache with better memory management."""
    
    def __init__(self):
        self.databases = {}
        self.lock = threading.RLock()
        self.max_cache_size = 3  # Limit cache size
        
    def get_or_create_database(self, sq_model_path: str, use_gpu: bool) -> Tuple[oefastrocs.OEShapeDatabase, oeshape.OEShapeQuery, oefastrocs.OEShapeDatabaseOptions]:
        """Enhanced database creation with better error handling."""
        with self.lock:
            key = f"{_get_file_hash(sq_model_path)}_{use_gpu}"
            
            # Cleanup old entries if cache too large
            if len(self.databases) >= self.max_cache_size:
                self._cleanup_oldest()
            
            if key in self.databases:
                db, query, opts = self.databases[key]
                return db, query, opts
            
            try:
                # Create query
                query = oeshape.OEShapeQuery()
                model_path = sq_model_path if isinstance(sq_model_path, str) else sq_model_path[0]
                
                if not oeshape.OEReadShapeQuery(model_path, query):
                    raise ValueError(f"Invalid shape query file: {model_path}")
                
                # Create options with vROCS-compatible settings
                opts = oefastrocs.OEShapeDatabaseOptions()
                # Force ROCS mode for all configurations to match CLI behavior
                try:
                    opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_ROCS)
                    # Create proper OEColorForceField object matching vROCS reference
                    color_ff = oeshape.OEColorForceField()
                    color_ff.Init(oeshape.OEColorFFType_ImplicitMillsDeanNoRings)  # Match vROCS exactly
                    opts.SetColorForceField(color_ff)
                    # Set limit to return only best hit per molecule (matches -besthits 1)
                    opts.SetLimit(1)  # Return only best hit per molecule
                    # Enable color optimization for better scoring
                    opts.SetColorOptimization(True)
                    # Set score cutoff to ensure all scores are returned
                    opts.SetScoreCutoff(-1.0)
                except AttributeError:
                    pass
                
                # Create database with color force field directly (vROCS approach)
                try:
                    # Try creating database with color force field directly
                    db = oefastrocs.OEShapeDatabase(color_ff)
                    db.SetNumOpenThreads(1)  # Always use single thread per worker
                except:
                    # Fallback to standard database creation
                    db = oefastrocs.OEShapeDatabase()
                    db.SetNumOpenThreads(1)  # Always use single thread per worker
                
                # Store in cache
                self.databases[key] = (db, query, opts)
                return db, query, opts
                
            except Exception as e:
                print(f"Error creating shape database: {e}")
                raise
                
    def _cleanup_oldest(self):
        """Remove oldest cache entry."""
        if self.databases:
            oldest_key = next(iter(self.databases))
            del self.databases[oldest_key]
            gc.collect()
            
    def close_all(self):
        """Enhanced cleanup of all database resources."""
        with self.lock:
            self.databases.clear()
            gc.collect()

# Global enhanced database cache
_SHAPE_DB_CACHE = EnhancedShapeDatabaseCache()

# ===============================================================================
# Enhanced scoring with database reuse
# ===============================================================================

def _score_molecules_with_database(isomers: List[oechem.OEMol], title2parent: Dict[str, int],
                                sq_model: str, use_gpu: bool) -> Dict[int, float]:
    """Enhanced scoring with better database management."""
    
    if not isomers:
        return {}
        
    try:
        # Get cached database components
        db, query, opts = _SHAPE_DB_CACHE.get_or_create_database(sq_model, use_gpu)
    except Exception as e:
        print(f"Error creating shape database: {e}")
        return {}
    
    scores: Dict[int, float] = {}
    
    # Enhanced database preparation with limits
    with _tmpdir(prefix="rocs_mols", use_cache=not use_gpu, 
                cache_key=_get_file_hash(sq_model) if not use_gpu else None) as td:
        
        database_path = os.path.join(td, "confs.oeb")
        
        # Limit molecules for database preparation
        limited_isomers = isomers[:min(len(isomers), _MAX_MOLECULES_PER_DB)]
        
        mdb = _prepare_molecule_database_enhanced(limited_isomers, database_path, use_gpu)
        if not mdb:
            return {}
        
        
        # Debug: List all molecules in the database and their titles
        # print("DEBUG: Molecules in database:")
        # for i in range(min(mdb.NumMols(), 20)):  # Show first 20 molecules
        #     title = mdb.GetTitle(i)
        #     print(f"  mol_idx={i}, title='{title}', parent_mapping={title2parent.get(title, 'NOT_FOUND')}")
        
        # Create fresh database for scoring with vROCS-compatible approach
        # Use the color force field from cached options to create database
        try:
            # Get color force field from cached options
            cached_color_ff = opts.GetColorForceField()
            if cached_color_ff:
                fresh_db = oefastrocs.OEShapeDatabase(cached_color_ff)
            else:
                fresh_db = oefastrocs.OEShapeDatabase()
        except (AttributeError, TypeError):
            fresh_db = oefastrocs.OEShapeDatabase()
        
        fresh_db.SetNumOpenThreads(1)
        
        # Update options for this database size
        try:
            opts.SetLimit(len(limited_isomers) * 2)  # Allow more hits than molecules
            opts.SetScoreCutoff(-1.0)  # Ensure all scores are returned
        except AttributeError:
            pass
        
        try:
            if not fresh_db.Open(mdb):
                return {}
            
            # Process scores with vROCS-compatible iteration
            score_count = 0
            max_scores = min(1000, len(limited_isomers) * 2)  # Reasonable limit
            
            for sc in fresh_db.GetSortedScores(query, opts):
                if score_count >= max_scores:
                    break
                    
                mol_idx = sc.GetMolIdx()
                mol_title = mdb.GetTitle(mol_idx)
                parent = title2parent.get(mol_title)
                
                if parent is not None:
                    tc = sc.GetTanimotoCombo()
                    scores[parent] = max(tc, scores.get(parent, 0.0))
                    
                score_count += 1
                
        except Exception as e:
            print(f"Error during scoring: {e}")
        finally:
            # Enhanced cleanup
            fresh_db = None
            mdb = None
            gc.collect()
        
    return scores

def _prepare_molecule_database_enhanced(molecules: List[oechem.OEMol], output_path: str, use_gpu: bool = False) -> oechem.OEMolDatabase:
    """Enhanced molecule database preparation with better error handling."""
    if not molecules:
        return None
        
    try:
        print(f"Preparing {len(molecules)} molecules for {'GPU' if use_gpu else 'CPU'} processing...")
        
        with oechem.oemolostream() as ofs:
            if use_gpu:
                oechem.OEPRECompress(ofs)
            
            if not ofs.open(output_path):
                print(f"Error: Could not open output file {output_path}")
                return None
                
            processed_count = 0
            
            for mol in molecules:
                try:
                    if use_gpu:
                        # Prepare for FastROCS
                        oefastrocs.OEPrepareFastROCSMol(mol)
                        half_mol = oechem.OEMol(mol, oechem.OEMCMolType_HalfFloatCartesian)
                        oechem.OEWriteMolecule(ofs, half_mol)
                    else:
                        oechem.OEWriteMolecule(ofs, mol)
                    processed_count += 1
                except Exception as e:
                    print(f"Warning: Failed to prepare molecule: {e}")
                    continue
        
        # Create and open database
        mdb = oechem.OEMolDatabase()
        if not mdb.Open(output_path):
            return None
            
        return mdb
        
    except Exception as e:
        print(f"Error preparing molecule database: {e}")
        return None

# ===============================================================================
# Enhanced worker initialization
# ===============================================================================

def _init_worker_enhanced(worker_id=None):
    """Enhanced worker initialization with better isolation."""
    global _WORKER_INITIALIZED
    
    if _WORKER_INITIALIZED:
        return
    
    try:
        # Enhanced environment setup
        os.environ["OE_SILENT"] = "true"
        os.environ["OPENEYE_SILENT"] = "true"
        
        # Set strict error handling
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)
        
        # Initialize memory pool for this worker
        _initialize_oe_memory_pool()
        
        # Disable GPU in worker processes explicitly
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["OPENEYE_GPU_MEMORY_LIMIT"] = "0"
        
        # Set CPU affinity if available
        if PSUTIL_AVAILABLE and worker_id is not None:
            try:
                process = psutil.Process()
                cpu_count = psutil.cpu_count(logical=True)
                if cpu_count > 1:
                    # More conservative CPU assignment
                    cpu_id = worker_id % max(1, cpu_count - 2)
                    process.cpu_affinity([cpu_id])
            except Exception:
                pass
        
        # Force cleanup
        gc.collect()
        _WORKER_INITIALIZED = True
        
    except Exception as e:
        print(f"Warning: Worker initialization failed: {e}")

# ===============================================================================
# Enhanced batch scoring with better error handling
# ===============================================================================

def _score_batch(batch: Tuple[List[str], List[int]],
                sq_model: str,
                max_iso: int,
                max_rot: int,
                max_heavy: int,
                use_gpu: bool = True,
                max_confs: int = 10,
                worker_id: int = None) -> Dict[int, float]:
    """Enhanced batch scoring with comprehensive error handling and limits."""
    
    smiles, idxs = batch
    
    # Initialize worker with enhanced settings
    _init_worker_enhanced(worker_id)
    
    # More conservative parameter limits
    safe_max_iso = min(max_iso, 2)
    safe_max_confs = min(max_confs, 5)
    safe_max_rot = min(max_rot, 15)
    safe_max_heavy = min(max_heavy, 40)
    
    try:
        # Check available memory before processing
        if PSUTIL_AVAILABLE:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 85:
                print(f"Warning: High memory usage ({memory_percent}%), reducing batch size")
                # Process only half the batch
                mid_point = len(smiles) // 2
                smiles = smiles[:mid_point]
                idxs = idxs[:mid_point]
        
        # Enhanced GPU check
        if use_gpu:
            try:
                is_gpu_ready = oefastrocs.OEFastROCSIsGPUReady()
                if not is_gpu_ready:
                    print("Warning: GPU not ready, falling back to CPU")
                    use_gpu = False
            except Exception:
                print("Warning: GPU check failed, using CPU")
                use_gpu = False
        
        # Prepare molecules with enhanced limits
        isomers, title2parent = _prepare_molecules_for_scoring(
            smiles, idxs, safe_max_iso, safe_max_rot, safe_max_heavy, use_gpu, safe_max_confs
        )
        
        if not isomers:
            return {}
        
        # Score molecules with enhanced error handling
        try:
            scores = _score_molecules_with_database(isomers, title2parent, sq_model, use_gpu)
        except Exception as e:
            print(f"Error in scoring: {e}")
            scores = {}
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        scores = {}
    finally:
        # Enhanced cleanup
        if 'isomers' in locals():
            isomers.clear()
        if 'title2parent' in locals():
            title2parent.clear()
        gc.collect()
    
    return scores

# ------------------------------------------------------------------------------
#  Scorer class
# ------------------------------------------------------------------------------

class OpenEyeScorer(Scorer):
    """
    FastROCS-based scorer. If input molecules have conformers, this scorer will match vROCS GUI speed by skipping OMEGA and using persistent in-memory databases. OMEGA is only called for SMILES or OEMol with no conformers. Uses OEPrepareFastROCSMol and OEPRECompress for best performance.
    """

    def __init__(self,
                sq_model_path: str | List[str],
                use_gpu: bool = True,
                max_isomers: int = 4,         # Aligned with base_rocs.py
                max_rot_bonds: int = 15,      # Aligned with base_rocs.py  
                max_heavy_atoms: int = 35,    # Aligned with base_rocs.py
                max_conformers: int = 200,    # Aligned with base_rocs.py CLI default
                cpu_processes: int | None = None,
                use_existing_conformers_always: bool = True):
        """
        Initialize the OpenEye FastROCS scorer with enhanced stability.

        Parameters
        ----------
        sq_model_path : str or List[str]
            Path to the ROCS query file (.sq file) or list of paths for multiple models
        use_gpu : bool, optional
            Whether to use GPU acceleration if available (default: True)
        max_isomers : int, optional
            Maximum number of isomers to enumerate per molecule (default: 4, aligned with base_rocs.py)
        max_rot_bonds : int, optional
            Maximum number of rotatable bonds to consider (default: 15, aligned with base_rocs.py)
        max_heavy_atoms : int, optional
            Maximum number of heavy atoms to process (default: 35, aligned with base_rocs.py)
        max_conformers : int, optional
            Maximum number of conformers to generate per molecule (default: 200, aligned with base_rocs.py CLI)
        cpu_processes : int | None, optional
            Number of CPU processes to use if not using GPU. If None, will use
            conservative calculation based on available resources.
            Ignored when GPU mode is active.
        use_existing_conformers_always : bool, optional
            Whether to use existing 3D conformers when available instead of generating new ones.
            When True, molecules with valid 3D coordinates will skip OMEGA conformer generation
            for improved performance (default: True)
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

        # Store the primary model for compatibility with older code
        self.sq_model = self.sq_models[0]
        self.max_iso = max_isomers
        self.max_rot = max_rot_bonds
        self.max_heavy = max_heavy_atoms
        self.max_confs = max_conformers
        self.use_existing_conformers_always = use_existing_conformers_always

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

        # Validate the query file and prepare
        try:
            # Create database options with ROCS CLI-compatible settings for better correlation
            opts = oefastrocs.OEShapeDatabaseOptions()
            # Force ROCS mode for all configurations to match CLI behavior
            try:
                opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_ROCS)
            except AttributeError:
                pass
            
            # Apply ROCS CLI-compatible settings for better correlation
            try:
                opts.SetRandomSeed(42)  # Fixed seed for reproducibility
                opts.SetSinglePrecision(False)  # Use double precision like VROCS
                opts.SetNumInertialStarts(4)  # Match VROCS default
                opts.SetLimit(1)  # Return only best hit per molecule (matches -besthits 1)
                opts.SetScoreCutoff(-1.0)  # Return all scores (no cutoff) - CRITICAL FIX
                # Create proper OEColorForceField object with vROCS-compatible enum
                color_ff = oeshape.OEColorForceField()
                color_ff.Init(oeshape.OEColorFFType_ImplicitMillsDeanNoRings)  # Match vROCS reference
                opts.SetColorForceField(color_ff)  # Match CLI -chemff parameter
                # Enable color optimization for better scoring
                opts.SetColorOptimization(True)
            except AttributeError:
                # Some options may not be available in all versions
                pass
                
            # Just validate the first model in the list
            primary_model = self.sq_models[0]
            query = oeshape.OEShapeQuery()
            if not oeshape.OEReadShapeQuery(primary_model, query):
                raise ValueError(f"Invalid shape query file: {primary_model}")
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
            pass    # ------------------------------------------------------------------
    #  Conformer deduplication methods (matching BaseROCSScorer)
    # ------------------------------------------------------------------
    
    def _extract_base_molecule_name(self, full_name: str) -> str:
        """
        Extract base molecule name from conformer-specific naming.
        
        Handles patterns like:
        - ZINC04617942_conf_0 -> ZINC04617942
        - mol_0+0_conf_5 -> mol_0
        - molecule_123_conformer_2 -> molecule_123
        - ZINC04617942 -> ZINC04617942 (unchanged)
        """
        import re
        
        # Pattern 1: _conf_N or _conformer_N
        pattern1 = r'(.+?)_conf(?:ormer)?_\d+$'
        match1 = re.match(pattern1, full_name, re.IGNORECASE)
        if match1:
            return match1.group(1)
        
        # Pattern 2: mol_X+Y_conf_N -> mol_X
        pattern2 = r'(mol_\d+)\+\d+_conf_\d+$'
        match2 = re.match(pattern2, full_name, re.IGNORECASE)
        if match2:
            return match2.group(1)
        
        # Pattern 3: Any general _N suffix where N is digits
        pattern3 = r'(.+?)_\d+$'
        match3 = re.match(pattern3, full_name)
        if match3:
            base_candidate = match3.group(1)
            # Only apply if it looks like a conformer pattern
            if any(keyword in full_name.lower() for keyword in ['conf', 'conformer']):
                return base_candidate
        
        # Return original name if no pattern matches
        return full_name
    
    def _select_best_conformers(self, results: Dict[str, float]) -> Dict[str, float]:
        """
        Select the best-scoring conformer for each base molecule name.
        
        Parameters
        ----------
        results : Dict[str, float]
            Dictionary mapping molecule names to scores
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping base molecule names to best scores
        """
        if not results:
            return {}
        
        # Group results by base molecule name
        base_molecule_groups = {}
        for mol_name, score in results.items():
            base_name = self._extract_base_molecule_name(mol_name)
            
            if base_name not in base_molecule_groups:
                base_molecule_groups[base_name] = []
            base_molecule_groups[base_name].append((mol_name, score))
        
        # Select best conformer for each base molecule
        best_conformers = {}
        for base_name, conformers in base_molecule_groups.items():
            # Find conformer with highest score
            best_conformer = max(conformers, key=lambda x: x[1])
            best_conformers[base_name] = best_conformer[1]  # Store the score
        
        return best_conformers
    
    def _ensure_consistent_naming(self, input_data) -> list:
        """
        Ensure consistent base molecule naming for input molecules.
        
        Parameters
        ----------
        input_data : list
            List of SMILES strings or OEMol objects
            
        Returns
        -------
        list
            List with consistent base molecule naming
        """
        if not input_data:
            return input_data
        
        result = []
        for item in input_data:
            if isinstance(item, str):
                # For SMILES strings, assume they're already properly named
                result.append(item)
            elif hasattr(item, 'GetTitle') and hasattr(item, 'SetTitle'):
                # For OEMol objects, ensure consistent naming
                current_title = item.GetTitle()
                base_name = self._extract_base_molecule_name(current_title)
                if base_name != current_title:
                    # Create a copy and update the title
                    mol_copy = oechem.OEMol(item)
                    mol_copy.SetTitle(base_name)
                    result.append(mol_copy)
                else:
                    result.append(item)
            else:
                result.append(item)
        
        return result

    # ------------------------------------------------------------------
    #  Public scoring methods
    # ------------------------------------------------------------------
    def getScores(self, mols, frags=None):
        if not isinstance(mols, list):
            mols = [mols]
        # FAST PATH: All OEMol with conformers (only if use_existing_conformers_always is True)
        if (self.use_existing_conformers_always and 
            all(hasattr(m, 'NumConfs') and m.NumConfs() > 0 for m in mols)):
            print("FAST PATH: Using all existing conformers, no OMEGA call.")
            temp_file = tempfile.NamedTemporaryFile(suffix=".oeb.gz", delete=False)
            temp_file.close()
            ofs = oechem.oemolostream()
            oechem.OEPRECompress(ofs)
            if ofs.open(temp_file.name):
                for i, mol in enumerate(mols):
                    oefastrocs.OEPrepareFastROCSMol(mol)
                    half_mol = oechem.OEMol(mol, oechem.OEMCMolType_HalfFloatCartesian)
                    oechem.OEWriteMolecule(ofs, half_mol)
                ofs.close()
            # Fix: Use database-based scoring for molecules with conformers
            # Create a dummy smiles list with proper length for database scoring
            dummy_smiles = [f"molecule_{i}" for i in range(len(mols))]
            
            # Fix: Use direct database scoring for molecules with conformers
            # Create molecule database from the temporary file
            mdb = oechem.OEMolDatabase()
            if not mdb.Open(temp_file.name):
                os.unlink(temp_file.name)
                return np.zeros(len(mols))
            
            # Score using FastROCS
            try:
                db, query, opts = _SHAPE_DB_CACHE.get_or_create_database(self.sq_model, self.use_gpu)
                
                # CRITICAL FIX: Ensure score cutoff is properly set for scoring
                try:
                    opts.SetScoreCutoff(-1.0)  # Ensure all scores are returned
                    opts.SetMaxHits(0)  # Return all hits
                except AttributeError:
                    pass
                
                # Create fresh database and copy configuration from cached options
                fresh_db = oefastrocs.OEShapeDatabase()
                fresh_db.SetNumOpenThreads(1)
                
                # Extract and apply the color force field from cached options if available
                try:
                    cached_color_ff = opts.GetColorForceField()
                    if cached_color_ff:
                        fresh_db = oefastrocs.OEShapeDatabase(cached_color_ff)
                        fresh_db.SetNumOpenThreads(1)
                except (AttributeError, TypeError):
                    pass
                
                scores_dict = {}
                if fresh_db.Open(mdb):
                    
                    score_count = 0
                    scores_iterator = fresh_db.GetSortedScores(query, opts)
                    
                    # Try to iterate and see if we get any scores at all
                    try:
                        for score in scores_iterator:
                            mol_idx = score.GetMolIdx()
                            # Map molecule index to original index
                            if mol_idx < len(mols):
                                tc = score.GetTanimotoCombo()
                                scores_dict[mol_idx] = max(tc, scores_dict.get(mol_idx, 0.0))
                                score_count += 1
                    except Exception as e:
                        print(f"Error during scoring iteration: {e}")
                
                fresh_db = None
                mdb = None
                
            except Exception as e:
                print(f"Error in FastROCS scoring: {e}")
                import traceback
                traceback.print_exc()
                scores_dict = {}
            
            os.unlink(temp_file.name)
            return np.array([scores_dict.get(i, 0.0) for i in range(len(mols))])
        # Otherwise, fallback to minimal OMEGA for SMILES or OEMol with no conformers
        print("FALLBACK: Some molecules lack conformers, using OMEGA only for those.")
        return self.__call__(mols)

    def _has_valid_3d_conformers(self, mol):
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

    # DrugEx explorers call the object itself
    def __call__(self, mols) -> np.ndarray:
        # accept OEMol / RDKit / SMILES
        if not isinstance(mols, list):
            mols = [mols]
            
        result = np.zeros(len(mols))
        
        # Check for mixed input types and handle OEMol objects with simple 3D conformer checking
        has_oemol_input = any(hasattr(mol, 'NumConfs') for mol in mols if mol is not None)
        
        if has_oemol_input:
            # Simple conformer processing - check if OEMol objects have valid 3D conformers
            processed_mols = []
            valid_indices = []
            
            for i, mol in enumerate(mols):
                if mol is None:
                    continue
                elif isinstance(mol, str):
                    if mol.strip():  # Valid SMILES
                        processed_mols.append(mol)
                        valid_indices.append(i)
                elif isinstance(mol, oechem.OEMol):
                    # Simple check: if molecule has conformers and 3D coordinates, use it as SMILES
                    if self._has_valid_3d_conformers(mol):
                        # Convert to SMILES for pipeline compatibility
                        smi = oechem.OECreateSmiString(mol)
                        if smi:
                            processed_mols.append(smi)
                            valid_indices.append(i)
                    else:
                        # Convert to SMILES and process normally
                        smi = oechem.OECreateSmiString(mol)
                        if smi:
                            processed_mols.append(smi)
                            valid_indices.append(i)
                elif RDKIT_AVAILABLE and hasattr(mol, "GetNumAtoms"):
                    smi = Chem.MolToSmiles(mol)
                    if smi:
                        processed_mols.append(smi)
                        valid_indices.append(i)
                else:
                    continue
            
            if not processed_mols:
                return result
            
            # Use the existing scoring pipeline with converted SMILES
            scores_array = self._score(processed_mols)
            
            # Map scores back to original indices
            for i, score_val in enumerate(scores_array):
                if i < len(valid_indices):
                    orig_idx = valid_indices[i]
                    result[orig_idx] = score_val
            
            return result
        
        # Standard processing fallback
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
        Handles multiple models by taking the maximum score for each molecule.
        """
        if not smiles:
            return np.zeros(0)
            
        # Use all available models
        active_models = self.sq_models
        
        # If we have multiple models, process them and take the maximum score
        if len(active_models) > 1:
            all_results = {}
            
            # Process each model and keep track of maximum scores
            for model in active_models:
                model_scores = self._score_with_single_model(smiles, model)
                
                # Update with max scores
                for idx, score in enumerate(model_scores):
                    all_results[idx] = max(score, all_results.get(idx, 0.0))
                    
                # Force garbage collection after each model
                gc.collect()
                
            # Convert dictionary to array
            max_scores = np.zeros(len(smiles))
            for idx, score in all_results.items():
                if 0 <= idx < len(smiles):
                    max_scores[idx] = score
                    
            return max_scores
        else:
            # For a single model, use the original scoring method
            return self._score_with_single_model(smiles, self.sq_model)
            
    def _score_with_single_model(self, smiles: List[str], sq_model: str) -> np.ndarray:
        """
        Score molecules with a single model.
        Original implementation of the scoring method.
        """
        if not smiles:
            return np.zeros(0)
            
        # Adjusted batch size calculation based on mode - Enhanced conservatism
        if self.use_gpu:
            # GPU mode - use smaller batches for stability  
            batch_size = 20  # Reduced from 30
        else:
            # CPU mode - use much smaller batches to prevent segfaults
            if len(smiles) <= 30:
                batch_size = 10  # Very small batches for small sets
            elif len(smiles) <= 100:
                batch_size = 5   # Extremely small for medium sets
            else:
                batch_size = 3   # Ultra-conservative for large sets
        
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
                
        # For CPU mode: use enhanced conservative approach for better stability
        else:
            results = {}
            
            # For small molecule sets, process sequentially to avoid multiprocessing overhead
            if len(smiles) <= 50:  # Reduced threshold from 60 to 50
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
                # For larger sets, use very limited multiprocessing for stability
                # Use only 1 worker to prevent segmentation faults
                ctx = mp.get_context("spawn")
                
                with ProcessPoolExecutor(max_workers=1,  # Ultra-conservative: single worker
                                         mp_context=ctx) as pool:
                    futures = []
                    for i, batch in enumerate(batches):
                        worker_id = 0  # Single worker
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
                            import traceback
                            traceback.print_exc()
        
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

def _get_file_hash(filepath):
    """Generate a simple hash for a file to use as cache key."""
    try:
        stat = os.stat(filepath)
        return f"{os.path.basename(filepath)}_{stat.st_size}_{int(stat.st_mtime)}"
    except (OSError, IOError):
        return os.path.basename(filepath)

# Cache for SMILES validation, significantly speeds up repeated checks
@lru_cache(maxsize=1000)
def _is_valid_smiles(smiles: str) -> bool:
    """Cached SMILES validation check."""
    if not smiles or not isinstance(smiles, str) or len(smiles) < 2:
        return False
        
    mol = oechem.OEMol()
    return bool(oechem.OESmilesToMol(mol, smiles))

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
def _get_omega_options(use_gpu: bool, max_confs: int = 200):
    """Get cached omega options for conformer generation."""
    omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts.SetMaxConfs(max_confs)
    
    # Configure GPU mode if available and requested
    omega_gpu_mode = False
    if use_gpu:
        try:
            if oeomega.OEOmegaIsGPUReady():
                omega_gpu_mode = True
                omegaOpts.GetTorDriveOptions().SetUseGPU(True)
                # Use ROCS CLI-compatible force field for better correlation
                from openeye import oeff
                omegaOpts.GetTorDriveOptions().SetForceField(oeff.OEMMFFSheffieldFFType_ImplicitMillsDean)
                omegaOpts.GetMolBuilderOptions().SetSampleHydrogens(False)
            else:
                omegaOpts.GetTorDriveOptions().SetUseGPU(False)
                omegaOpts.SetSampleHydrogens(True)
        except Exception as e:
            print(f"Warning: Error configuring Omega GPU mode: {e}")
            omegaOpts.GetTorDriveOptions().SetUseGPU(False)
            omegaOpts.SetSampleHydrogens(True)
    else:
        omegaOpts.GetTorDriveOptions().SetUseGPU(False)
        omegaOpts.SetSampleHydrogens(True)
    
    # Apply VROCS-compatible OMEGA settings
    omegaOpts.SetStrictStereo(False)
    omegaOpts.SetFromCT(True)
    omegaOpts.SetMaxConfs(max_confs)
    omegaOpts.SetFixRMS(True)
    omegaOpts.SetRMSThreshold(0.5)
    omegaOpts.SetEnumRing(True)
    omegaOpts.SetRotorOffset(False)
    
    return omegaOpts