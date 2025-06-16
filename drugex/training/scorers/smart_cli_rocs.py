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
import gc
import time
import multiprocessing as mp
from typing import Union, List, Dict, Optional, Tuple, Any
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import threading
from dataclasses import dataclass, field
from collections import OrderedDict

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

# ===============================================================================
# PERFORMANCE CONFIGURATION
# ===============================================================================

@dataclass
class ROCSPerformanceConfig:
    """High-performance ROCS configuration with caching and optimization"""
    
    # Cache settings - Conservative defaults for stability
    enable_conformer_cache: bool = True
    max_cache_size_mb: int = 512   # Reduced from 2048 to prevent corruption
    cache_cleanup_threshold: float = 0.6  # More aggressive cleanup
    
    # Performance settings  
    max_workers: int = 2  # Conservative for memory
    process_isolation_threshold: int = 500  # Use processes for large batches
    memory_monitoring_interval: int = 50  # Check every N molecules
    
    # Conformer generation limits
    max_conformers_cached: int = 50  # Reduced for caching
    max_isomers_cached: int = 2      # Reduced for caching  
    max_conformers_direct: int = 200  # Full generation when not cached
    max_isomers_direct: int = 4       # Full generation when not cached
    
    # Memory limits - More conservative for RL training
    memory_pressure_threshold: float = 0.75  # Cleanup at 75% RAM (reduced from 85%)
    process_memory_limit_mb: int = 2048      # 2GB per process (reduced from 4GB)
    
    # Timeouts and retries
    rocs_timeout_seconds: int = 300
    max_retry_attempts: int = 3
    
    # Debug settings (disabled for production)
    enable_performance_logging: bool = False
    enable_memory_logging: bool = False

# Global configuration instance
PERF_CONFIG = ROCSPerformanceConfig()

# ===============================================================================
# SMART CONFORMER CACHE
# ===============================================================================

class ConformerCache:
    """Thread-safe LRU cache for conformers with memory management and corruption detection"""
    
    def __init__(self, max_size_mb: int = 2048):
        self.max_size_mb = max_size_mb
        self.cache = OrderedDict()  # LRU cache
        self.cache_lock = threading.RLock()
        self.current_size_mb = 0
        self._stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'corruptions': 0}
        self._corruption_threshold = 5  # Max corruptions before disabling
        
        # Create cache directory
        self.cache_dir = Path(tempfile.gettempdir()) / "smart_rocs_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, smiles: str, max_conformers: int, max_isomers: int) -> str:
        """Generate cache key for SMILES with parameters"""
        key_data = f"{smiles}:{max_conformers}:{max_isomers}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def _estimate_molecule_size_mb(self, mol) -> float:
        """Estimate memory usage of molecule in MB"""
        if not mol:
            return 0.0
        # Rough estimate: atoms * conformers * bytes per atom
        atoms = mol.NumAtoms() if hasattr(mol, 'NumAtoms') else 50
        conformers = mol.GetMaxConfIdx() + 1 if hasattr(mol, 'GetMaxConfIdx') else 1
        return (atoms * conformers * 200) / (1024 * 1024)  # ~200 bytes per atom-conformer
        
    def _cleanup_cache(self, target_size_ratio: float = 0.7):
        """Remove oldest entries to reach target size"""
        if not self.cache:
            return
            
        target_size_mb = self.max_size_mb * target_size_ratio
        removed_count = 0
        
        with self.cache_lock:
            while self.current_size_mb > target_size_mb and self.cache:
                key, mol = self.cache.popitem(last=False)  # Remove oldest
                size_mb = self._estimate_molecule_size_mb(mol)
                self.current_size_mb -= size_mb
                removed_count += 1
                self._stats['evictions'] += 1
                
        # Cache cleanup completed silently
    
    def get(self, smiles: str, max_conformers: int, max_isomers: int) -> Optional[List]:
        """Get cached conformers for SMILES"""
        key = self._get_cache_key(smiles, max_conformers, max_isomers)
        
        with self.cache_lock:
            if key in self.cache:
                # Move to end (most recently used)
                mol_data = self.cache.pop(key)
                self.cache[key] = mol_data
                self._stats['hits'] += 1
                return mol_data
            else:
                self._stats['misses'] += 1
                return None
                
    def put(self, smiles: str, max_conformers: int, max_isomers: int, molecules: List) -> None:
        """Cache conformers for SMILES"""
        if not molecules:
            return
            
        key = self._get_cache_key(smiles, max_conformers, max_isomers)
        
        # Estimate size of new entry
        total_size_mb = sum(self._estimate_molecule_size_mb(mol) for mol in molecules)
        
        with self.cache_lock:
            # Check if we need cleanup
            if self.current_size_mb + total_size_mb > self.max_size_mb * PERF_CONFIG.cache_cleanup_threshold:
                self._cleanup_cache()
                
            # Add to cache if there's space
            if self.current_size_mb + total_size_mb <= self.max_size_mb:
                self.cache[key] = molecules
                self.current_size_mb += total_size_mb
                
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics with corruption detection"""
        with self.cache_lock:
            hit_rate = self._stats['hits'] / max(1, self._stats['hits'] + self._stats['misses'])
            
            # Detect corruption: impossible states
            is_corrupted = (
                (len(self.cache) == 0 and self.current_size_mb > 100) or  # No entries but large size
                (self.current_size_mb > self.max_size_mb * 1.1) or        # Size exceeds limit significantly
                (len(self.cache) > 10000)                                  # Unreasonable entry count
            )
            
            if is_corrupted:
                self._stats['corruptions'] += 1
                if self._stats['corruptions'] >= self._corruption_threshold:
                    self.clear()  # Auto-recovery
                    
            return {
                'size_mb': self.current_size_mb,
                'max_size_mb': self.max_size_mb,
                'entries': len(self.cache),
                'hit_rate': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'corrupted': is_corrupted
            }
    
    def clear(self):
        """Clear all cached data and reset corruption stats"""
        with self.cache_lock:
            self.cache.clear()
            self.current_size_mb = 0
            self._stats['corruptions'] = 0  # Reset corruption counter
            
    def __del__(self):
        """Cleanup cache directory on destruction"""
        try:
            if hasattr(self, 'cache_dir') and self.cache_dir.exists():
                shutil.rmtree(self.cache_dir, ignore_errors=True)
        except:
            pass

# Global conformer cache with thread safety lock
_CACHE_GLOBAL_LOCK = threading.RLock()
_CONFORMER_CACHE = ConformerCache(PERF_CONFIG.max_cache_size_mb)

# ===============================================================================
# ENHANCED MEMORY MANAGEMENT
# ===============================================================================

class MemoryManager:
    """Advanced memory monitoring and management"""
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get current memory usage information"""
        if not PSUTIL_AVAILABLE:
            return {'available_gb': 8.0, 'used_percent': 0.5}
            
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3), 
            'used_gb': memory.used / (1024**3),
            'used_percent': memory.percent / 100.0,
            'free_gb': memory.free / (1024**3)
        }
    
    @staticmethod
    def check_memory_pressure() -> bool:
        """Check if memory pressure is high"""
        info = MemoryManager.get_memory_info()
        return info['used_percent'] > PERF_CONFIG.memory_pressure_threshold
    
    @staticmethod
    def force_cleanup():
        """Aggressive memory cleanup"""
        _CONFORMER_CACHE.clear()
        gc.collect()
        try:
            if hasattr(oechem, 'OEClearMemory'):
                oechem.OEClearMemory()
        except:
            pass
    
    @staticmethod
    def log_memory_usage(context: str = ""):
        """Log current memory usage (silent in production)"""
        pass

# ===============================================================================
# PROCESS ISOLATION SYSTEM
# ===============================================================================

def _isolated_scoring_worker(args_tuple):
    """Worker function for process-isolated ROCS scoring"""
    smiles_chunk, query_files, config_dict, worker_id = args_tuple
    
    # Set process limits
    os.environ["OE_SILENT"] = "true"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    try:
        # Create temporary scorer with limited parameters
        scorer = SmartCLIROCSScorer(
            query_files=query_files,
            max_conformers=config_dict.get('max_conformers', 50),
            max_isomers=config_dict.get('max_isomers', 2),
            batch_size_limit=min(50, len(smiles_chunk)),
            show_progress=False,
            enable_caching=False  # No caching in workers
        )
        
        start_time = time.time()
        scores = scorer._score_molecules_cached(smiles_chunk)
        elapsed = time.time() - start_time
        
        return {
            'scores': scores,
            'worker_id': worker_id,
            'elapsed': elapsed,
            'molecule_count': len(smiles_chunk)
        }
        
    except Exception as e:
        return {
            'scores': np.zeros(len(smiles_chunk)),
            'worker_id': worker_id,
            'error': str(e),
            'molecule_count': len(smiles_chunk)
        }
    finally:
        # Aggressive cleanup in worker
        try:
            scorer.cleanup_resources()
            del scorer
        except:
            pass
        MemoryManager.force_cleanup()

class ProcessIsolationManager:
    """Manages process isolation for large batch scoring"""
    
    @staticmethod
    def should_use_isolation(molecule_count: int) -> bool:
        """Determine if process isolation should be used"""
        return molecule_count >= PERF_CONFIG.process_isolation_threshold
    
    @staticmethod
    def create_chunks(smiles_list: List[str], max_chunk_size: int = 100) -> List[List[str]]:
        """Split molecules into chunks for processing"""
        chunks = []
        for i in range(0, len(smiles_list), max_chunk_size):
            chunk = smiles_list[i:i + max_chunk_size]
            chunks.append(chunk)
        return chunks
    
    @staticmethod
    def score_with_isolation(smiles_list: List[str], query_files: List[str], 
                           config_dict: Dict[str, Any]) -> np.ndarray:
        """Score molecules using process isolation"""
        
        if PERF_CONFIG.enable_performance_logging:
            print(f"Using process isolation for {len(smiles_list)} molecules")
            
        # Create chunks
        chunk_size = min(100, len(smiles_list) // PERF_CONFIG.max_workers + 1)
        chunks = ProcessIsolationManager.create_chunks(smiles_list, chunk_size)
        
        # Prepare worker arguments
        worker_args = []
        for i, chunk in enumerate(chunks):
            worker_args.append((chunk, query_files, config_dict, i))
        
        # Process chunks in parallel
        all_scores = []
        
        try:
            with ProcessPoolExecutor(max_workers=PERF_CONFIG.max_workers) as executor:
                results = list(executor.map(_isolated_scoring_worker, worker_args))
                
            # Combine results
            for result in results:
                if 'error' in result:
                    print(f"Worker {result['worker_id']} error: {result['error']}")
                all_scores.extend(result['scores'])
                
            if PERF_CONFIG.enable_performance_logging:
                total_time = sum(r.get('elapsed', 0) for r in results)
                total_mols = sum(r.get('molecule_count', 0) for r in results)
                if total_time > 0:
                    print(f"Process isolation: {total_mols} molecules, "
                          f"{total_mols/total_time:.1f} mol/sec")
                          
        except Exception as e:
            print(f"Process isolation failed: {e}")
            # Fallback to direct scoring
            return np.zeros(len(smiles_list))
            
        return np.array(all_scores[:len(smiles_list)])  # Ensure correct length

def _check_memory_pressure():
    """Quick memory pressure check."""
    if PSUTIL_AVAILABLE:
        try:
            return psutil.virtual_memory().percent > 80.0
        except:
            pass
    return False

def _force_cleanup():
    """Aggressive cleanup to free memory."""
    gc.collect()
    try:
        if hasattr(oechem, 'OEClearMemory'):
            oechem.OEClearMemory()
    except:
        pass

@contextmanager
def _managed_tmpdir():
    """Managed temporary directory with guaranteed cleanup."""
    path = tempfile.mkdtemp(prefix="smart_rocs_")
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except:
            pass

def _score_chunk_worker(mols_data, query_files, max_conformers, max_isomers):
    """Worker function for isolated chunk processing."""
    os.environ["OE_SILENT"] = "true"
    
    try:
        # Create temporary scorer with conservative limits
        scorer = SmartCLIROCSScorer(
            query_files=query_files,
            max_conformers=min(max_conformers, 50),
            max_isomers=min(max_isomers, 2),
            batch_size_limit=50,
            show_progress=False
        )
        return scorer._score_chunk_directly(mols_data)
    finally:
        _force_cleanup()

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
                 score_type: str = "TanimotoCombo",  # NEW: From cli_base_rocs.py
                 max_conformers: int = None,  # Now uses smart defaults
                 max_isomers: int = None,     # Now uses smart defaults  
                 max_heavy_atoms: int = 35,
                 max_rotatable_bonds: int = 15,
                 shape_only: bool = False,
                 optimize: bool = True,       # NEW: From cli_base_rocs.py
                 color_optimize: bool = True, # NEW: From cli_base_rocs.py
                 color_force_field: str = "ImplicitMillsDean",  # NEW: From cli_base_rocs.py
                 use_gpu: bool = False,
                 rocs_binary: str = "rocs",   # Renamed from binary_path for consistency
                 binary_path: str = None,     # NEW: From cli_base_rocs.py (for compatibility)
                 output_file: str = None,     # NEW: From cli_base_rocs.py
                 max_retry_attempts: int = 3,
                 batch_size_limit: int = None,  # Now auto-determined
                 show_progress: bool = False,
                 enable_caching: bool = True,    # NEW: Enable conformer caching
                 performance_config: Optional[ROCSPerformanceConfig] = None):
        
        super().__init__()
        if not OE_AVAILABLE:
            raise ImportError("OpenEye toolkits required")
        
        # Convert to list and validate
        self.query_files = [query_files] if isinstance(query_files, str) else list(query_files)
        
        # Use performance config or global default
        self.config = performance_config or PERF_CONFIG
        
        # NEW: Parameters from cli_base_rocs.py
        self.score_type = score_type
        self.optimize = optimize
        self.color_optimize = color_optimize
        self.color_force_field = color_force_field
        self.binary_path = binary_path or rocs_binary  # Support both parameter names
        self.output_file = output_file
        
        # Smart parameter defaults based on caching
        if enable_caching:
            self.max_conformers = max_conformers or self.config.max_conformers_cached
            self.max_isomers = max_isomers or self.config.max_isomers_cached
            self.batch_size_limit = batch_size_limit or 200  # Smaller for cached
        else:
            self.max_conformers = max_conformers or self.config.max_conformers_direct
            self.max_isomers = max_isomers or self.config.max_isomers_direct
            self.batch_size_limit = batch_size_limit or 1000  # Larger for direct
        
        # Store configuration
        self.enable_caching = enable_caching
        self.max_heavy_atoms = max_heavy_atoms
        self.max_rotatable_bonds = max_rotatable_bonds
        self.use_gpu = use_gpu
        self.shape_only = shape_only
        self.rocs_binary = rocs_binary  # Keep for backward compatibility
        self.max_retry_attempts = max_retry_attempts
        self.show_progress = show_progress
        self.batch_size_limit = batch_size_limit
        
        self._validate_query_files()
        
        # Validate binary path exists (from cli_base_rocs.py)
        if not shutil.which(self.binary_path):
            raise FileNotFoundError(f"ROCS binary not found: {self.binary_path}")
        
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
        
    def __call__(self, mols, frags=None) -> np.ndarray:
        """DrugEx interface - score molecules and return numpy array (from cli_base_rocs.py)"""
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
            elif hasattr(mol, 'GetNumAtoms'):  # RDKit molecule
                try:
                    from rdkit import Chem
                    smi = Chem.MolToSmiles(mol)
                    smiles_list.append(smi if smi else "")
                except:
                    smiles_list.append("")
            else:
                smiles_list.append("")
        
        return self.getScores(smiles_list)
        
    def getScores(self, mols, frags=None) -> np.ndarray:
        """High-performance ROCS scoring with thread-safe caching and corruption protection"""
        if not mols:
            return np.zeros(0)
            
        # Global thread lock for RL training safety
        with _CACHE_GLOBAL_LOCK:
            timer = oechem.OEWallTimer() if self.show_progress else None
            num_input_mols = len(mols)
            
            if self.show_progress:
                print(f"Starting ROCS scoring for {num_input_mols} molecules...")
                MemoryManager.log_memory_usage("before scoring")
            
            # Check cache health before processing
            if self.enable_caching:
                cache_stats = _CONFORMER_CACHE.get_stats()
                if cache_stats.get('corrupted', False):
                    if self.show_progress:
                        print("Cache corruption detected - clearing cache")
                    _CONFORMER_CACHE.clear()
            
            # Convert to SMILES list for uniform processing
            smiles_list = self._convert_to_smiles(mols)
            
            # Check if we should use process isolation
            if ProcessIsolationManager.should_use_isolation(num_input_mols):
                config_dict = {
                    'max_conformers': self.max_conformers,
                    'max_isomers': self.max_isomers,
                    'enable_caching': False  # No caching in isolated processes
                }
                result_scores = ProcessIsolationManager.score_with_isolation(
                    smiles_list, self.query_files, config_dict
                )
            else:
                # Use in-process scoring with caching
                result_scores = self._score_molecules_cached(smiles_list)
            
            if self.show_progress:
                if timer and timer.Elapsed() > 2.0:
                    print(f"ROCS scoring completed in {timer.Elapsed():.1f}s")
                MemoryManager.log_memory_usage("after scoring")
                
                # Show cache statistics
                if self.enable_caching:
                    cache_stats = _CONFORMER_CACHE.get_stats()
                    print(f"Cache: {cache_stats['hit_rate']:.1%} hit rate, "
                          f"{cache_stats['entries']} entries, "
                          f"{cache_stats['size_mb']:.1f}MB")
            
            return result_scores
    
    def _score_molecules_cached(self, smiles_list: List[str]) -> np.ndarray:
        """Score molecules using intelligent conformer caching"""
        num_mols = len(smiles_list)
        result_scores = np.zeros(num_mols)
        
        # Process molecules with caching
        processed_mols = []
        cache_hits = 0
        cache_misses = 0
        
        for i, smi in enumerate(smiles_list):
            if not smi or not smi.strip():
                continue
                
            # Check cache first (if enabled)
            cached_conformers = None
            if self.enable_caching:
                cached_conformers = _CONFORMER_CACHE.get(
                    smi.strip(), self.max_conformers, self.max_isomers
                )
                
            if cached_conformers is not None:
                # Use cached conformers
                cache_hits += 1
                for mol in cached_conformers:
                    mol_copy = oechem.OEMol(mol)
                    mol_copy.SetTitle(f"mol_{i}")
                    processed_mols.append(mol_copy)
            else:
                # Generate new conformers
                cache_misses += 1
                new_conformers = self._generate_conformers_for_smiles(smi.strip(), i)
                
                if new_conformers:
                    processed_mols.extend(new_conformers)
                    
                    # Cache the conformers (if enabled)
                    if self.enable_caching:
                        _CONFORMER_CACHE.put(
                            smi.strip(), self.max_conformers, self.max_isomers, new_conformers
                        )
            
            # Memory pressure check
            if i % self.config.memory_monitoring_interval == 0:
                if MemoryManager.check_memory_pressure():
                    MemoryManager.log_memory_usage("memory pressure detected")
                    MemoryManager.force_cleanup()
        
        if self.show_progress and (cache_hits + cache_misses) > 0:
            hit_rate = cache_hits / (cache_hits + cache_misses)
            print(f"Conformer generation: {cache_hits} cached, {cache_misses} generated "
                  f"(hit rate: {hit_rate:.1%})")
        
        if not processed_mols:
            return result_scores
            
        # Score the processed molecules
        try:
            scores_dict = self._score_with_retry(processed_mols)
            
            # Map scores back to original indices
            for mol_key, score in scores_dict.items():
                try:
                    if mol_key.startswith('mol_'):
                        idx = int(mol_key.split('_')[1])
                        if 0 <= idx < num_mols:
                            result_scores[idx] = max(result_scores[idx], score)
                except (ValueError, IndexError):
                    continue
                    
        except Exception:
            return result_scores
            
        return result_scores
    
    def _generate_conformers_for_smiles(self, smiles: str, mol_idx: int) -> List[oechem.OEMol]:
        """Generate conformers for a single SMILES string"""
        conformers = []
        
        try:
            mol = oechem.OEMol()
            if not oechem.OESmilesToMol(mol, smiles):
                return conformers
                
            mol.SetTitle(f"mol_{mol_idx}")
            
            # Generate isomers
            isomers = self._enumerate_isomers(mol)
            
            # Generate conformers for each isomer
            omega = self._create_fresh_omega()
            
            try:
                for isomer in isomers:
                    try:
                        if omega(isomer):
                            # Extract all conformers
                            for conf in isomer.GetConfs():
                                conf_mol = oechem.OEMol(conf)
                                conf_mol.SetTitle(f"mol_{mol_idx}")
                                conformers.append(conf_mol)
                    except Exception:
                        # Fallback: use isomer without conformers
                        isomer.SetTitle(f"mol_{mol_idx}")
                        conformers.append(isomer)
                        
            finally:
                omega = None  # Cleanup
                
        except Exception:
            return conformers
            
        return conformers
    
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
        
        # Create fresh Omega instance (no caching!)
        omega = self._create_fresh_omega()
        
        # Progress tracking for conformer generation
        valid_molecules = len([s for s in smiles_list if s.strip()])
        dots = None
        if self.show_progress and valid_molecules > 50:
            print("Generating conformers...")
            dots = oechem.OEThreadedDots(100, 50, "molecules")
        
        try:
            for i, smi in enumerate(smiles_list):
                if not smi or not smi.strip():
                    continue
                
                # Memory check every 50 molecules
                if i % 50 == 0 and _check_memory_pressure():
                    _force_cleanup()
                    
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
        
        finally:
            # Clean up Omega instance
            omega = None
            _force_cleanup()
                
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
        
    def _create_fresh_omega(self):
        """Create a fresh Omega instance with proven parameters"""
        opts = oeomega.OEOmegaOptions()
        # Use conservative conformer limits
        opts.SetMaxConfs(min(self.max_conformers, 100))  # Cap at 100 vs unlimited
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
        
        with _managed_tmpdir() as tmpdir:
            try:
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
                # Build molecule index mapping from the scores array
                for i, score in enumerate(scores_array):
                    # The scores array is indexed by original molecule index (from parsing)
                    mol_key = f"mol_{i}"
                    scores[mol_key] = score
            except Exception as e:
                print(f"Error in single query scoring: {e}")
            finally:
                _force_cleanup()
                    
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
        
        # Debug output removed for production
        
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
        """Build ROCS command matching cli_base_rocs.py exactly"""
        # Extract the directory from output_file
        output_dir = os.path.dirname(output_file) or "."
        
        cmd = [
            self.binary_path, 
            "-query", query_file, 
            "-dbase", input_file,
            "-report", "one", 
            "-reportfile", output_file,
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
            
        return cmd
        
    def _parse_results(self, output_file: str, num_mols: int) -> np.ndarray:
        """Parse ROCS output and return numpy array"""
        scores = np.zeros(num_mols)
        
        if not os.path.exists(output_file):
            return scores
            
        try:
            import pandas as pd
            df = pd.read_csv(output_file, sep='\t')
            
            if df.empty or 'Name' not in df.columns or self.score_type not in df.columns:
                return scores
            
            for _, row in df.iterrows():
                name = row.get('Name', '')
                if name.startswith('mol_'):
                    try:
                        idx = int(name.split('_')[1].split('+')[0])
                        if 0 <= idx < num_mols:
                            score_value = float(row.get(self.score_type, 0.0))
                            scores[idx] = max(scores[idx], score_value)
                    except (ValueError, IndexError):
                        continue
                        
        except Exception:
            pass
            
        return scores
        
    def getKey(self) -> str:
        """Return scorer identifier"""
        return "ROCS"

    def __del__(self):
        """Proper cleanup when scorer is destroyed."""
        try:
            _force_cleanup()
        except Exception:
            pass

    def cleanup_resources(self):
        """Enhanced cleanup including cache management"""
        try:
            if self.enable_caching:
                _CONFORMER_CACHE.clear()
            MemoryManager.force_cleanup()
        except Exception:
            pass
    
    def configure_performance(self, 
                            enable_caching: Optional[bool] = None,
                            cache_size_mb: Optional[int] = None,
                            memory_threshold: Optional[float] = None,
                            batch_size_limit: Optional[int] = None) -> None:
        """Configure performance settings dynamically"""
        global PERF_CONFIG
        
        if enable_caching is not None:
            self.enable_caching = enable_caching
            
        if cache_size_mb is not None:
            PERF_CONFIG.max_cache_size_mb = cache_size_mb
            _CONFORMER_CACHE.max_size_mb = cache_size_mb
            
        if memory_threshold is not None:
            PERF_CONFIG.memory_pressure_threshold = memory_threshold
            
        if batch_size_limit is not None:
            self.batch_size_limit = batch_size_limit
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        cache_stats = _CONFORMER_CACHE.get_stats() if self.enable_caching else {}
        memory_info = MemoryManager.get_memory_info()
        
        return {
            'caching_enabled': self.enable_caching,
            'cache_stats': cache_stats,
            'memory_info': memory_info,
            'config': {
                'max_conformers': self.max_conformers,
                'max_isomers': self.max_isomers,
                'batch_size_limit': self.batch_size_limit,
                'process_isolation_threshold': PERF_CONFIG.process_isolation_threshold
            }
        }

    def _process_large_batch(self, mols) -> np.ndarray:
        """Process large batches by splitting into smaller chunks"""
        num_mols = len(mols)
        result_scores = np.zeros(num_mols)
        
        # Check memory before starting
        if _check_memory_pressure():
            _force_cleanup()
        
        # Conservative batch size based on memory pressure
        batch_size = 25 if _check_memory_pressure() else 50
        total_chunks = (num_mols + batch_size - 1) // batch_size
        
        if self.show_progress:
            print(f"Processing {num_mols} molecules in {total_chunks} chunks")
        
        # Process in chunks
        for chunk_idx, start_idx in enumerate(range(0, num_mols, batch_size)):
            end_idx = min(start_idx + batch_size, num_mols)
            chunk = mols[start_idx:end_idx]
            
            if self.show_progress:
                print(f"  Chunk {chunk_idx + 1}/{total_chunks}")
            
            try:
                chunk_scores = self._score_chunk_with_cleanup(chunk)
                result_scores[start_idx:end_idx] = chunk_scores
            except Exception:
                continue
            finally:
                # Cleanup after each chunk
                _force_cleanup()
                
        return result_scores
    
    def _score_chunk_with_cleanup(self, chunk):
        """Score chunk with cleanup or process isolation."""
        # Use process isolation if memory pressure is high or chunk is large
        if len(chunk) > 100 or _check_memory_pressure():
            return self._score_chunk_isolated(chunk)
        else:
            return self._score_chunk_directly(chunk)

    def _score_chunk_isolated(self, chunk):
        """Score chunk in isolated process."""
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
            future = executor.submit(
                _score_chunk_worker, 
                chunk, 
                self.query_files,
                self.max_conformers,
                self.max_isomers
            )
            try:
                return future.result()
            except Exception:
                return np.zeros(len(chunk))
    
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
                # Check memory before each attempt
                if _check_memory_pressure():
                    _force_cleanup()
                    time.sleep(1)  # Brief pause for system recovery
                
                # Multi-query scoring (no nested tmpdir - let individual methods handle it)
                if len(self.query_files) == 1:
                    scores_dict = self._score_single_query(processed_mols, self.query_files[0])
                else:
                    scores_dict = self._score_multi_query(processed_mols)
                    
                    if scores_dict:
                        return scores_dict
                        
            except Exception:
                _force_cleanup()
                if attempt < self.max_retry_attempts:
                    time.sleep(1)  # Brief pause
            finally:
                # Always cleanup after each attempt
                _force_cleanup()
        
        return {} 