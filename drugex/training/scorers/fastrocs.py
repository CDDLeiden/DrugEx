#!/usr/bin/env python3
"""
FastROCS‑based scorer used by DrugEx‑ROCS.

Key points
----------
✓  GPU : single‑process only (FastROCS limitation)  
✓  CPU : fork‑server workers, each with CUDA disabled  
✓  OpenEye memory pool set before forking  
✓  Interface compatible with the original code (`getScores`, `__call__`)
"""

from __future__ import annotations
import os, tempfile, multiprocessing as mp
import sys
from functools import partial
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple
import warnings

import numpy as np
from openeye import oechem, oeomega, oeshape, oefastrocs, oeff
from drugex.training.scorers.interfaces import Scorer

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
#  Generic helpers
# ------------------------------------------------------------------------------

def _init_worker():
    """Initialiser for every fork‑server worker."""
    oechem.OESetMemPoolMode(oechem.OEMemPoolMode_System)
    os.environ["OE_SILENT"] = "true"


@contextmanager
def _tmpdir(prefix="fastrocs_"):
    path = tempfile.mkdtemp(prefix=prefix)
    try:
        yield path
    finally:        # best‑effort cleanup
        for root, _, files in os.walk(path, topdown=False):
            for f in files:
                try: os.remove(os.path.join(root, f))
                except Exception: pass
        try: os.rmdir(path)
        except Exception: pass


# ------------------------------------------------------------------------------
#  Per‑molecule utilities
# ------------------------------------------------------------------------------

_BAD_ATOMS = {'Au','Ag','Al','As','Be','Bi','Ce','Dy','Eu'}

def filter_molecules(smiles_list: List[str], max_rot: int = 10, max_heavy: int = 30) -> List[Tuple[str, bool]]:
    """
    Filter molecules for FastROCS processing, returning list of (smiles, is_valid) tuples.
    Performs comprehensive checks for problematic molecules in one pass.
    """
    results = []
    
    # Define SMARTS patterns for problematic structures
    problem_patterns = [
        '[S+]', '[n+]', '[N+](=[O-])', '[#7,#16]~[#7,#16]',
        '[C,c]#[C,c]', '[#6]=[#6]=[#6]', '[r3]'
    ]
    
    # Compile SMARTS patterns
    compiled_patterns = []
    for pattern in problem_patterns:
        pat = oechem.OESubSearch()
        if pat.Init(pattern):
            compiled_patterns.append(pat)
    
    for smi in smiles_list:
        if not smi or not isinstance(smi, str):
            results.append((smi, False))
            continue
            
        mol = oechem.OEMol()
        if not oechem.OESmilesToMol(mol, smi):
            results.append((smi, False))
            continue
        
        # Quick check for problematic atoms
        if any(oechem.OEGetAtomicSymbol(a.GetAtomicNum()) in _BAD_ATOMS for a in mol.GetAtoms()):
            results.append((smi, False))
            continue
            
        # Check rotatable bonds and heavy atoms
        if (oechem.OECount(mol, oechem.OEIsRotor()) > max_rot or
            oechem.OECount(mol, oechem.OEIsHeavy()) > max_heavy):
            results.append((smi, False))
            continue
            
        # Check against all patterns
        atom_count = mol.NumAtoms()
        if (any(pat.SingleMatch(mol) for pat in compiled_patterns) or
            atom_count > 100 or atom_count < 3):
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

def _enumerate_isomers(mol: oechem.OEMol, max_centers=4, max_iso=4):
    opts = oeomega.OEFlipperOptions()
    opts.SetMaxCenters(max_centers)
    for i, conf in enumerate(oeomega.OEFlipper(mol, opts)):
        if i == max_iso:
            break
        iso = oechem.OEMol(conf)
        iso.SetTitle(f"{mol.GetTitle()}+{i}")
        yield iso


def _score_batch(batch: Tuple[List[str], List[int]],
                 sq_model: str,
                 max_iso: int,
                 max_rot: int,
                 max_heavy: int,
                 use_gpu: bool = True) -> Dict[int, float]:
    """
    Process and score a batch of molecules.
    Combines filtering, conformer generation, and scoring in one efficient function.
    """
    smiles, idxs = batch
    
    # Apply unified filtering to all molecules in batch
    filtered_data = []
    filter_results = filter_molecules(smiles, max_rot, max_heavy)
    
    for (smi, is_valid), idx in zip(filter_results, idxs):
        if is_valid:
            filtered_data.append((smi, idx))
    
    if not filtered_data:
        return {}
    
    title2parent: Dict[str, int] = {}
    isomers: List[oechem.OEMol] = []

    # Configure conformer generation once for all molecules
    omega = oeomega.OEOmega()
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
    
    omega.SetOptions(omegaOpts)
    omega.SetMaxConfs(10)
    
    # Generate conformers for all filtered molecules
    for s, idx in filtered_data:
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, s)
        mol.SetTitle(str(idx))
        
        for iso in _enumerate_isomers(mol, max_iso):
            omega(iso)
            for conf in iso.GetConfs():
                confmol = oechem.OEMol(conf)
                title2parent[confmol.GetTitle()] = idx
                isomers.append(confmol)

    if not isomers:
        return {}

    # Score molecules using FastROCS
    scores: Dict[int, float] = {}
    with _tmpdir() as td:
        sdf = os.path.join(td, "confs.sdf")
        with oechem.oemolostream(sdf) as ofs:
            for m in isomers:
                oechem.OEWriteMolecule(ofs, m)

        # Create shape DB and query
        mdb = oechem.OEMolDatabase()
        if not mdb.Open(sdf):
            return {}
            
        db = oefastrocs.OEShapeDatabase()
        db.SetNumOpenThreads(1)
        if not db.Open(mdb):
            return {}

        query = oeshape.OEShapeQuery()
        if not oeshape.OEReadShapeQuery(sq_model, query):
            return {}

        # Get scores
        opts = oefastrocs.OEShapeDatabaseOptions()
        for sc in db.GetSortedScores(query, opts):
            dbmol = oechem.OEMol()
            mdb.GetMolecule(dbmol, sc.GetMolIdx())
            parent = title2parent.get(dbmol.GetTitle())
            if parent is not None:
                tc = sc.GetTanimotoCombo()
                scores[parent] = max(tc, scores.get(parent, 0.0))
    
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

        oechem.OESetMemPoolMode(oechem.OEMemPoolMode_System)
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
        """Unified scoring method for both GPU and CPU modes."""
        if not smiles:
            return np.zeros(0)

        # Process in batches
        idxs = list(range(len(smiles)))
        batch_size = 30  # Fixed reasonable batch size
        batches = [(smiles[i:i+batch_size], idxs[i:i+batch_size])
                  for i in range(0, len(smiles), batch_size)]
        
        # GPU mode uses simple single-process scoring
        if self.use_gpu:
            results = {}
            for batch in batches:
                batch_results = _score_batch(
                    batch, self.sq_model, self.max_iso, 
                    self.max_rot, self.max_heavy, True
                )
                results.update(batch_results)
        # CPU mode uses multiprocessing
        else:
            ctx = mp.get_context("forkserver")
            with ProcessPoolExecutor(max_workers=self.cpu_procs,
                                    mp_context=ctx,
                                    initializer=_init_worker) as pool:
                fn = partial(_score_batch,
                            sq_model=self.sq_model,
                            max_iso=self.max_iso,
                            max_rot=self.max_rot,
                            max_heavy=self.max_heavy,
                            use_gpu=False)  # Always false in workers
                batch_results = list(pool.map(fn, batches))
                
                # Combine results
                results = {}
                for d in batch_results:
                    results.update(d)
        
        # Convert dictionary to array
        out = np.zeros(len(smiles))
        for k, v in results.items():
            out[k] = v
        return out

    # ------------------------------------------------------------------
    def getKey(self):
        return "ROCS"
