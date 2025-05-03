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
import os, gc, time, tempfile, multiprocessing as mp
from functools import partial
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
from openeye import oechem, oeomega, oeshape, oefastrocs
from drugex.training.scorers.interfaces import Scorer

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


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
                 max_heavy: int) -> Dict[int, float]:
    smiles, idxs = batch
    title2parent: Dict[str,int] = {}
    isomers: List[oechem.OEMol] = []

    # ---------- build conformers ---------------------------------------------
    # use Omega to generate 3D conformers for each stereoisomer
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(10)
    omega.SetStrictStereo(False)
    for s, idx in zip(smiles, idxs):
        mol = oechem.OEMol()
        if not oechem.OESmilesToMol(mol, s):
            continue
        if (any(oechem.OEGetAtomicSymbol(a.GetAtomicNum()) in _BAD_ATOMS for a in mol.GetAtoms()) or
            oechem.OECount(mol, oechem.OEIsRotor()) > max_rot or
            oechem.OECount(mol, oechem.OEIsHeavy()) > max_heavy):
            continue
        mol.SetTitle(str(idx))
        for iso in _enumerate_isomers(mol, max_iso):
            # add hydrogens and generate conformers
            # oechem.OEAddExplicitHydrogens(iso)
            omega(iso)
            for conf in iso.GetConfs():
                confmol = oechem.OEMol(conf)
                title2parent[confmol.GetTitle()] = idx
                isomers.append(confmol)

    if not isomers:
        return {}

    with _tmpdir() as td:
        sdf = os.path.join(td, "confs.sdf")
        with oechem.oemolostream(sdf) as ofs:
            for m in isomers:
                oechem.OEWriteMolecule(ofs, m)

        # create shape DB
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

        opts = oefastrocs.OEShapeDatabaseOptions()
        # limit to number of available conformers to prevent warnings
        # opts.SetLimit(len(isomers))

        out: Dict[int,float] = {}
        for sc in db.GetSortedScores(query, opts):
            dbmol = oechem.OEMol()
            mdb.GetMolecule(dbmol, sc.GetMolIdx())
            parent = title2parent.get(dbmol.GetTitle())
            if parent is not None:
                tc = sc.GetTanimotoCombo()
                out[parent] = max(tc, out.get(parent, 0.0))
        return out


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
        if not smiles:
            return np.zeros(0)

        if self.use_gpu:
            return self._score_gpu(smiles)
        else:
            return self._score_cpu(smiles)

    def _score_gpu(self, smiles: List[str]) -> np.ndarray:
        batch = (smiles, list(range(len(smiles))))
        res = _score_batch(batch, self.sq_model,
                           self.max_iso, self.max_rot, self.max_heavy)
        out = np.zeros(len(smiles))
        for k,v in res.items():
            out[k] = v
        return out

    def _score_cpu(self, smiles: List[str]) -> np.ndarray:
        idxs = list(range(len(smiles)))
        batch_size = 30
        batches = [(smiles[i:i+batch_size], idxs[i:i+batch_size])
                   for i in range(0, len(smiles), batch_size)]

        ctx = mp.get_context("forkserver")
        with ProcessPoolExecutor(max_workers=self.cpu_procs,
                                 mp_context=ctx,
                                 initializer=_init_worker) as pool:
            fn = partial(_score_batch,
                         sq_model=self.sq_model,
                         max_iso=self.max_iso,
                         max_rot=self.max_rot,
                         max_heavy=self.max_heavy)
            results = list(pool.map(fn, batches))

        out = np.zeros(len(smiles))
        for d in results:
            for k,v in d.items():
                out[k] = v
        return out

    # ------------------------------------------------------------------
    def getKey(self):
        return "ROCS"
