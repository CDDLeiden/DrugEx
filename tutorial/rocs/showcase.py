# %% [markdown]
# # DrugEx ROCS RNN Tutorial
# 
# ## Overview
# Comprehensive end-to-end workflow for ROCS-based molecular generation using DrugEx, mirroring the Sequence-RNN tutorial with ROCS scoring objectives.
# 
# ## Workflow
# 1. **Transfer Learning**: Fine-tune pretrained RNN on CCR dataset
# 2. **Reinforcement Learning**: Train models across multiple conformer/scorer backends:
#    - OpenEye (Omega conformers + ROCS CLI)
#    - RDKit (aggregate + supermolecule)
#    - CDPKit (aggregate + supermolecule)
# 3. **Sampling & Evaluation**: Generate molecules and cross-validate across backends
# 
# ## Key Components
# - **Conformer Generators**: Omega, RDKit, CDPKit
# - **ROCS Scorers**: CLI-based, RDKit-based, CDPKit-based variants
# - **Evaluation Metrics**: Shape/color similarity, scaffold analysis, novelty assessment
# - **Cross-Backend Validation**: Native scoring comparison across all backends
# 
# ## Prerequisites
# - Python 3.10 with DrugEx
# - RDKit
# - OpenEye toolkits + ROCS CLI
# - CDPKit
# - CCR dataset and ROCS query files
# 
# ## Output
# Trained RL agents for each backend combination with comprehensive evaluation of generated molecule sets including scaffold diversity, novelty metrics, and cross-backend scoring consistency.

# %% [markdown]
# ## Environment Setup

# %%
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
import pandas as pd
from IPython.display import display, HTML
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from drugex.training.generators.sequence_rnn import SequenceRNN
from drugex.training.scorers.conformer_generators import (
    OmegaConformerGenerator,
    RDKitConformerGenerator,
    CDPKitConformerGenerator,
    OE_AVAILABLE as OMEGA_AVAILABLE,
    CDPL_AVAILABLE as CDPKIT_AVAILABLE,
)
from drugex.training.scorers.cli_rocs import CLIROCSScorer, OE_AVAILABLE as ROCS_AVAILABLE
from drugex.training.scorers.rdkit_rocs import (
    RDKitAggregateScorer,
    RDKitSupermoleculeScorer,
)
from drugex.training.scorers.cdpkit_rocs import (
    CDPKitROCSAggregateScorer,
    CDPKitROCSSupermoleculeScorer,
)
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers.sequence_explorer import SequenceExplorer
from drugex.training.monitors import FileMonitor
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.corpus.corpus import SequenceCorpus
from drugex.data.processing import CorpusEncoder, RandomTrainTestSplitter
from drugex.data.datasets import SmilesDataSet
from drugex.logs import logger

NOTEBOOK_NAME = "DrugEx_ROCS_RNN_Showcase.ipynb"


def locate_notebook_dir() -> Path:
    """Resolve the directory containing this notebook independent of CWD."""
    candidates: List[Path] = []
    try:
        nb_path = Path(__file__).resolve()
        candidates.append(nb_path.parent)
    except NameError:
        pass

    cwd = Path.cwd().resolve()
    candidates.append(cwd)

    env_dir = os.environ.get("DRUGEX_NOTEBOOK_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    candidates.append(cwd / "tutorial" / "rocs")

    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue
        if (candidate / NOTEBOOK_NAME).exists():
            return candidate

    for parent in cwd.parents:
        candidate = (parent / "tutorial" / "rocs").resolve()
        if (candidate / NOTEBOOK_NAME).exists():
            return candidate

    return cwd


NOTEBOOK_DIR = locate_notebook_dir()


def locate_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists():
            return path
    return start


REPO_ROOT = locate_repo_root(NOTEBOOK_DIR)
DATA_ROOT = NOTEBOOK_DIR / "rocs_rl_ccr"
RESOURCES_DIR = DATA_ROOT / "rdkit_cdpkit"

CCR_DATA_PATH = RESOURCES_DIR / "CCR_HUMAN_AL.tsv"

RDKit_REF_SDF = RESOURCES_DIR / "CCR2_reference_ligands.sdf"
ROCS_QUERY = RDKit_REF_SDF
ROCS_SQ_QUERY = RESOURCES_DIR / "model3-4_v1.sq"

SUPER_SDF = RESOURCES_DIR / "supermol_123.sdf"

PRETRAINED_DIRS = [
    REPO_ROOT / "tutorial" / "data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT",
    REPO_ROOT / "data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT",
    NOTEBOOK_DIR / "../data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT",
]
PRETRAINED_VOCAB = "Papyrus05.5_smiles_rnn_PT.vocab"
PRETRAINED_PKG = "Papyrus05.5_smiles_rnn_PT.pkg"

FINETUNE_DIR = DATA_ROOT / "finetune_runs"
FINETUNE_CHECKPOINT = DATA_ROOT / "finetune.pkg"
FINETUNE_VOCAB = DATA_ROOT / "finetune.vocab"

print(f"Notebook directory: {NOTEBOOK_DIR}")
print(f"Repository root: {REPO_ROOT}")
print(f"Data root: {DATA_ROOT}")

# %% [markdown]
# ## Resource Checklist

# %%
resources = [
    ("Omega toolkit", OMEGA_AVAILABLE, None),
    ("ROCS CLI", shutil.which("rocs") is not None, shutil.which("rocs")),
    ("CDPKit (CDPL)", CDPKIT_AVAILABLE, None),
    ("CCR dataset", CCR_DATA_PATH.exists(), CCR_DATA_PATH),
    ("ROCS query SDF", ROCS_QUERY.exists(), ROCS_QUERY),
    ("ROCS query SQ", ROCS_SQ_QUERY.exists(), ROCS_SQ_QUERY),
    ("RDKit reference SDF", RDKit_REF_SDF.exists(), RDKit_REF_SDF),
    ("Supermolecule SDF", SUPER_SDF.exists(), SUPER_SDF),
    ("Finetune checkpoint", FINETUNE_CHECKPOINT.exists(), FINETUNE_CHECKPOINT),
    ("Finetune vocab", FINETUNE_VOCAB.exists(), FINETUNE_VOCAB),
]

resource_df = pd.DataFrame(
    [
        {
            "resource": name,
            "available": bool(flag),
            "path": str(path) if path else "",
        }
        for name, flag, path in resources
    ]
)
resource_df



# %% [markdown]
# ## Data Preparation

# %%
if not CCR_DATA_PATH.exists():
    raise FileNotFoundError(f"CCR dataset missing: {CCR_DATA_PATH}")

ccr_df = pd.read_csv(CCR_DATA_PATH, sep="	").dropna(subset=["SMILES"])
SAMPLE_SIZE = 128
candidate_smiles = ccr_df["SMILES"].head(SAMPLE_SIZE).tolist()


def embed_reference(mol: Chem.Mol) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    if status != 0:
        return None
    AllChem.UFFOptimizeMolecule(mol)
    return mol


@lru_cache(maxsize=1)
def load_rdkit_references() -> List[Chem.Mol]:
    refs: List[Chem.Mol] = []
    if not RDKit_REF_SDF.exists():
        return refs
    suppl = Chem.SDMolSupplier(str(RDKit_REF_SDF), removeHs=False)
    for mol in suppl:
        if mol is None:
            continue
        if mol.GetNumConformers() == 0:
            mol = embed_reference(mol)
        if mol is not None and mol.GetNumConformers() > 0:
            refs.append(mol)
    return refs


RDKit_REFERENCE_MOLS = load_rdkit_references()
print(f"Loaded {len(candidate_smiles)} candidate SMILES")
print(f"RDKit reference molecules available: {len(RDKit_REFERENCE_MOLS)}")

# %% [markdown]
# ## Transfer Learning Setup
# 
# Fine-tunes a pretrained RNN model on the CCR dataset to adapt it for domain-specific molecular generation.

# %%
RUN_TRANSFER = True
TRANSFER_CONFIG = {
    "epochs": 120,
    "train_split": 0.9,
    "chunk_size": 500,
    "batch_size": 256,
    "max_valid_size": 1000,
    "shuffle": True,
    "n_proc": 25,
}


def find_pretrained_dir() -> Path:
    for candidate in PRETRAINED_DIRS:
        vocab = candidate / PRETRAINED_VOCAB
        pkg = candidate / PRETRAINED_PKG
        if vocab.exists() and pkg.exists():
            return candidate
    raise FileNotFoundError("Papyrus pretrained checkpoint not found in configured locations")


@lru_cache(maxsize=1)
def load_vocabulary(update=False) -> VocSmiles:
    if FINETUNE_VOCAB.exists():
        return VocSmiles.fromFile(str(FINETUNE_VOCAB), encode_frags=False, max_len=100, min_len=10)
    pretrained_dir = find_pretrained_dir()
    return VocSmiles.fromFile(str(pretrained_dir / PRETRAINED_VOCAB), encode_frags=False, max_len=100, min_len=10)


def prepare_finetune_loaders(config: Dict) -> tuple:
    voc = load_vocabulary()
    smiles_all = ccr_df["SMILES"].dropna().tolist()
    splitter = RandomTrainTestSplitter(1.0 - config["train_split"], config["max_valid_size"], shuffle=config["shuffle"])
    smiles_train, smiles_valid = splitter(smiles_all)

    data_dir = DATA_ROOT / "encoded_rnn"
    data_dir.mkdir(parents=True, exist_ok=True)

    encoder = CorpusEncoder(
        SequenceCorpus,
        {
            "vocabulary": voc,
            "update_voc": False,
            "throw": True,
        },
        n_proc=config["n_proc"],
        chunk_size=config["chunk_size"],
    )

    train_collector = SmilesDataSet(data_dir / "ccr_train.tsv", rewrite=True)
    valid_collector = SmilesDataSet(data_dir / "ccr_valid.tsv", rewrite=True)

    encoder.apply(smiles_train, collector=train_collector)
    encoder.apply(smiles_valid, collector=valid_collector)

    train_loader = train_collector.asDataLoader(batch_size=config["batch_size"])
    valid_loader = valid_collector.asDataLoader(batch_size=config["batch_size"])
    return voc, train_loader, valid_loader


def run_transfer_learning(config: Dict, execute: bool = False) -> pd.DataFrame:
    records: List[Dict[str, str]] = []
    voc = load_vocabulary()
    pretrained_dir = find_pretrained_dir()
    pretrained_pkg = pretrained_dir / PRETRAINED_PKG

    if not execute:
        status = "skipped (set RUN_TRANSFER=True to execute)"
        if FINETUNE_CHECKPOINT.exists():
            status = "using cached finetune checkpoint"
        records.append(
            {
                "status": status,
                "checkpoint": str(FINETUNE_CHECKPOINT),
                "vocabulary": str(FINETUNE_VOCAB if FINETUNE_VOCAB.exists() else pretrained_dir / PRETRAINED_VOCAB),
                "epochs": config["epochs"],
            }
        )
        return pd.DataFrame(records)

    logger.setLevel("ERROR")
    voc, train_loader, valid_loader = prepare_finetune_loaders(config)

    model = SequenceRNN(voc=voc, is_lstm=True)
    model.loadStatesFromFile(str(pretrained_pkg))

    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    monitor_prefix = FINETUNE_DIR / f"finetune_ep{config['epochs']}"
    monitor = FileMonitor(str(monitor_prefix), save_smiles=True, reset_directory=True)

    model.fit(train_loader, valid_loader, epochs=config["epochs"], monitor=monitor)

    torch_state_path = FINETUNE_CHECKPOINT
    model_state = model.state_dict()
    torch.save(model_state, torch_state_path)
    voc.toFile(str(FINETUNE_VOCAB))

    records.append(
        {
            "status": "finetune completed",
            "checkpoint": str(FINETUNE_CHECKPOINT),
            "vocabulary": str(FINETUNE_VOCAB),
            "epochs": config["epochs"],
        }
    )
    return pd.DataFrame(records)


# %%
transfer_summary = run_transfer_learning(TRANSFER_CONFIG, execute=RUN_TRANSFER)
transfer_summary

# %% [markdown]
# ## Reinforcement Learning Backends

# %% [markdown]
# ## Reinforcement Learning Backends
# 
# Each backend bundles a conformer generator, one or more scorers, threshold
# settings, and an RL configuration. Set per-backend flags in `RL_EXECUTION` to
# rerun training; otherwise existing checkpoints are used.

# %%
from re import T


@dataclass
class BackendSpec:
    label: str
    library: str
    builder: Callable[[], tuple]
    thresholds: List[float]
    rl_dir: Path
    agent_pkg: Path
    monitor_prefix: str
    run_key: str
    scalers: Optional[List[SmoothClippedScore]] = None
    notes: Optional[str] = None


RL_EXECUTION = {
    "openeye": True,
    "rdkit_aggregate": True,
    "rdkit_supermol": True,
    "cdpkit_aggregate": True,
    "cdpkit_supermol": True,
}

RL_DEFAULTS = {
    "epochs": 30,
    "batch_size": 256,
    "epsilon": 0.1,
    "n_samples": 130,
    "reload_interval": 25,
    "criteria": "desired_ratio",
}


def build_openeye_backend():
    if not OMEGA_AVAILABLE:
        return None, "OpenEye toolkits not available"
    if not ROCS_AVAILABLE:
        return None, "OpenEye ROCS Python bindings not available"
    if shutil.which("rocs") is None:
        return None, "ROCS CLI binary not found in PATH"
    
    # Check for available query files
    query_files = {}
    if ROCS_QUERY.exists():
        query_files["CCR2_SDF"] = str(ROCS_QUERY)
    if ROCS_SQ_QUERY.exists():
        query_files["Model3-4_SQ"] = str(ROCS_SQ_QUERY)
    
    if not query_files:
        return None, f"No ROCS query files found. Checked: {ROCS_QUERY}, {ROCS_SQ_QUERY}"

    generator = OmegaConformerGenerator(max_conformers=40, show_progress=False)
    scorer = CLIROCSScorer(
        conformer_generator=generator,
        # Use both SDF and .sq queries; CLIROCSScorer will return max score across all queries
        query_files=query_files,
        score_type="TanimotoCombo",
        show_progress=False,
    )
    return [scorer], "ready"


def build_rdkit_aggregate():
    if not RDKit_REFERENCE_MOLS:
        return None, "No RDKit reference molecules"
    generator = RDKitConformerGenerator(max_conformers=40, show_progress=False)
    scorer = RDKitAggregateScorer(
        conformer_generator=generator,
        reference_mols=RDKit_REFERENCE_MOLS,
        show_progress=False,
    )
    return [scorer], "ready"


def build_rdkit_supermol():
    if not SUPER_SDF.exists():
        return None, f"Supermolecule SDF missing: {SUPER_SDF}"
    generator = RDKitConformerGenerator(max_conformers=40, show_progress=False)
    scorer = RDKitSupermoleculeScorer(
        conformer_generator=generator,
        supermol_file=str(SUPER_SDF),
        show_progress=False,
    )
    return [scorer], "ready"


def build_cdpkit_aggregate():
    if not CDPKIT_AVAILABLE:
        return None, "CDPKit not available"
    if not RDKit_REF_SDF.exists():
        return None, f"Reference SDF missing: {RDKit_REF_SDF}"
    generator = CDPKitConformerGenerator(
        max_conformers=40,  
        max_centers=4,
        timeout=3600,      # seconds
        min_rmsd=0.5,       # Ångströms
        energy_window=20.0, # kcal/mol
        show_progress=False
    )
    scorer = CDPKitROCSAggregateScorer(
        conformer_generator=generator,
        reference_mols=[str(RDKit_REF_SDF)],
        show_progress=False,
    )
    return [scorer], "ready"


def build_cdpkit_supermol():
    if not CDPKIT_AVAILABLE:
        return None, "CDPKit not available"
    if not SUPER_SDF.exists():
        return None, f"Supermolecule SDF missing: {SUPER_SDF}"
    generator = CDPKitConformerGenerator(
        max_conformers=40,  
        max_centers=4,
        timeout=3600,      # seconds
        min_rmsd=0.5,       # Ångströms
        energy_window=20.0, # kcal/mol
        show_progress=False
    )
    scorer = CDPKitROCSSupermoleculeScorer(
        conformer_generator=generator,
        supermol_file=str(SUPER_SDF),
        show_progress=False,
    )
    return [scorer], "ready"



BACKENDS: Dict[str, BackendSpec] = {
    "openeye": BackendSpec(
        label="OpenEye (Omega + ROCS)",
        library="OpenEye",
        builder=build_openeye_backend,
        thresholds=[0.35],
        rl_dir=DATA_ROOT / "rl_runs" / "openeye_rocs",
        agent_pkg=DATA_ROOT / "rl_runs" / "openeye_rocs" / "openeye_rocs.pkg",
        monitor_prefix="openeye_rocs",
        run_key="openeye",
    ),
    "rdkit_aggregate": BackendSpec(
        label="RDKit Aggregate",
        library="RDKit",
        builder=build_rdkit_aggregate,
        thresholds=[0.35],
        rl_dir=DATA_ROOT / "rl_runs" / "rdkit_aggregate",
        agent_pkg=DATA_ROOT / "rl_runs" / "rdkit_aggregate" / "rdkit_aggregate.pkg",
        monitor_prefix="rdkit_aggregate",
        run_key="rdkit_aggregate",
    ),
    "rdkit_supermol": BackendSpec(
        label="RDKit Supermolecule",
        library="RDKit",
        builder=build_rdkit_supermol,
        thresholds=[0.25],
        rl_dir=DATA_ROOT / "rl_runs" / "rdkit_supermol",
        agent_pkg=DATA_ROOT / "rl_runs" / "rdkit_supermol" / "rdkit_supermol.pkg",
        monitor_prefix="rdkit_supermol",
        run_key="rdkit_supermol",
    ),
    "cdpkit_aggregate": BackendSpec(
        label="CDPKit Aggregate",
        library="CDPKit",
        builder=build_cdpkit_aggregate,
        thresholds=[0.35],
        rl_dir=DATA_ROOT / "rl_runs" / "cdpkit_aggregate",
        agent_pkg=DATA_ROOT / "rl_runs" / "cdpkit_aggregate" / "cdpkit_aggregate.pkg",
        monitor_prefix="cdpkit_aggregate",
        run_key="cdpkit_aggregate",
    ),
    "cdpkit_supermol": BackendSpec(
        label="CDPKit Supermolecule",
        library="CDPKit",
        builder=build_cdpkit_supermol,
        thresholds=[0.25],
        rl_dir=DATA_ROOT / "rl_runs" / "cdpkit_supermol",
        agent_pkg=DATA_ROOT / "rl_runs" / "cdpkit_supermol" / "cdpkit_supermol.pkg",
        monitor_prefix="cdpkit_supermol",
        run_key="cdpkit_supermol",
    ),
}

LABEL_TO_KEY = {spec.label: key for key, spec in BACKENDS.items()}



# %% [markdown]
# ### RL Utilities

# %% [markdown]
# We follow the multi-objective pattern from `Sequence-RNN.ipynb` by pairing each shape scorer
# with a synthetic accessibility (SA) observer. The environment automatically attaches a
# `Property('SA')` scorer and clips it with `SmoothClippedScore(lower_x=5.0, upper_x=3.0)`,
# ensuring reinforcement learning optimizes both TanimotoCombo and synthesizability.
# 

# %%
def load_sequence_model(checkpoint: Path, voc: VocSmiles) -> SequenceRNN:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    model = SequenceRNN(voc=voc, is_lstm=True)
    model.loadStatesFromFile(str(checkpoint))
    return model


def ensure_rl_dirs(spec: BackendSpec):
    spec.rl_dir.mkdir(parents=True, exist_ok=True)


SA_THRESHOLD = 0.1

def build_environment_and_modifier(scorers: List, thresholds: List[float]):
    """Attach default modifiers and a synthetic accessibility objective."""
    expanded_scorers = list(scorers)
    expanded_thresholds = list(thresholds)

    for scorer in expanded_scorers:
        if scorer.getModifier() is None:
            scorer.setModifier(SmoothClippedScore(upper_x=0.9, lower_x=0.4))

    def has_sa(scorer_list):
        for scorer in scorer_list:
            key = scorer.getKey()
            if isinstance(key, str) and key == "SA":
                return True
            if isinstance(key, list) and any(k == "SA" for k in key):
                return True
        return False

    if not has_sa(expanded_scorers):
        sa_scorer = Property("SA")
        sa_scorer.setModifier(SmoothClippedScore(lower_x=5.0, upper_x=3.0))
        expanded_scorers.append(sa_scorer)

    # Count scorer keys (not scorers) to match DrugExEnvironment logic
    scorer_keys = []
    for scorer in expanded_scorers:
        key = scorer.getKey()
        if isinstance(key, list):
            scorer_keys.extend(key)
        else:
            scorer_keys.append(key)
    
    # Ensure we have enough thresholds for all scorer keys
    while len(expanded_thresholds) < len(scorer_keys):
        expanded_thresholds.append(SA_THRESHOLD)

    env = DrugExEnvironment(
        scorers=expanded_scorers,
        thresholds=expanded_thresholds,
        reward_scheme=ParetoCrowdingDistance(),
    )
    return env


def run_rl_backend(spec: BackendSpec, execute: bool, rl_config: Dict, voc: VocSmiles) -> Dict:
    ensure_rl_dirs(spec)
    scorers, status = spec.builder()
    if scorers is None:
        return {
            "backend": spec.label,
            "status": status,
            "agent_path": str(spec.agent_pkg),
        }

    if not execute and spec.agent_pkg.exists():
        return {
            "backend": spec.label,
            "status": "using cached agent",
            "agent_path": str(spec.agent_pkg),
        }

    if not FINETUNE_CHECKPOINT.exists():
        return {
            "backend": spec.label,
            "status": "finetune checkpoint missing",
            "agent_path": str(spec.agent_pkg),
        }

    env = build_environment_and_modifier(scorers, spec.thresholds)
    agent = load_sequence_model(FINETUNE_CHECKPOINT, voc)
    mutate = load_sequence_model(FINETUNE_CHECKPOINT, voc)
    crover = load_sequence_model(FINETUNE_CHECKPOINT, voc)

    explorer = SequenceExplorer(
        agent=agent,
        env=env,
        mutate=mutate,
        crover=crover,
        batch_size=rl_config["batch_size"],
        epsilon=rl_config["epsilon"],
        n_samples=rl_config["n_samples"],
    )

    monitor = FileMonitor(
        str(spec.rl_dir / spec.monitor_prefix),
        save_smiles=True,
        reset_directory=True,
    )

    explorer.fit(
        epochs=rl_config["epochs"],
        monitor=monitor,
        reload_interval=rl_config.get("reload_interval", 25),
        criteria=rl_config.get("criteria", "desired_ratio"),
    )

    return {
        "backend": spec.label,
        "status": "rl completed",
        "agent_path": str(spec.agent_pkg),
    }


# %%
voc = load_vocabulary()
rl_records = []
for key, spec in BACKENDS.items():
    execute = RL_EXECUTION.get(spec.run_key, False)
    summary = run_rl_backend(spec, execute, RL_DEFAULTS, voc)
    rl_records.append(summary)
rl_df = pd.DataFrame(rl_records)
rl_df

# %% [markdown]
# ## Sampling & Cross-Backend Evaluation
# 
# Generates molecular samples from each trained RL agent to enable cross-backend validation. This systematic comparison across different conformer/scorer combinations (OpenEye, RDKit, CDPKit) quantifies backend consistency and identifies optimal molecular generation strategies for CCR ligands.

# %%
SAMPLE_CONFIG = {
    "per_backend": 1000,
    "batch_size": 256,
    "max_batches": 60,
}


def sample_unique_smiles(model: SequenceRNN, target_size: int, batch_size: int, max_batches: int) -> List[str]:
    if target_size <= 0:
        return []
    unique: List[str] = []
    seen: set[str] = set()
    for _ in range(max_batches):
        batch = model.sample(batch_size)
        for smi in batch:
            if smi not in seen:
                seen.add(smi)
                unique.append(smi)
                if len(unique) >= target_size:
                    return unique
    return unique


def load_agent_for_sampling(path: Path, voc: VocSmiles) -> Optional[SequenceRNN]:
    if not path.exists():
        return None
    model = SequenceRNN(voc=voc, is_lstm=True)
    model.loadStatesFromFile(str(path))
    return model


def collect_backend_samples(backends: Dict[str, BackendSpec], config: Dict, voc: VocSmiles) -> Dict[str, List[str]]:
    samples: Dict[str, List[str]] = {}
    for key, spec in backends.items():
        agent = load_agent_for_sampling(spec.agent_pkg, voc)
        if agent is None:
            continue
        
        # Sample unique SMILES using the agent
        smiles = sample_unique_smiles(
            agent, 
            config["per_backend"], 
            config["batch_size"], 
            config["max_batches"]
        )
        samples[spec.label] = smiles
    return samples

backend_samples = collect_backend_samples(BACKENDS, SAMPLE_CONFIG, voc)
{
    name: len(smiles) for name, smiles in backend_samples.items()
}

# %% [markdown]
# ## Cross-Backend Scoring Utilities
# 
# Provides functions to score molecular samples using any backend configuration, enabling systematic cross-validation of generated molecules across different conformer/scorer combinations.

# %%
def get_backend_spec_by_label(label: str):
    key = LABEL_TO_KEY.get(label)
    if key is None:
        return None, f"Unknown backend label: {label}"
    spec = BACKENDS[key]
    scorers, status = spec.builder()
    if scorers is None:
        return None, status
    return (spec, scorers), "ready"


def score_with_backend(label: str, smiles: List[str]) -> pd.Series:
    result, status = get_backend_spec_by_label(label)
    if result is None or not smiles:
        return pd.Series([np.nan] * len(smiles))
    spec, scorers = result
    env = build_environment_and_modifier(scorers, spec.thresholds)
    df_scores = env.getScores(smiles)
    keys = env.getScorerKeys()
    if len(keys) == 1:
        return pd.Series(df_scores[keys[0]])
    return df_scores[keys].mean(axis=1)


# %% [markdown]
# ### Sample Sanitization and Desired Filtering
# 
# Canonicalizes generated SMILES, removes unsanitizable molecules, and filters for valid-desired compounds according to each backend's scoring criteria. This preprocessing ensures downstream analyses operate on chemically valid, high-scoring molecular sets.

# %%
from rdkit.Chem import SanitizeMol

def sanitize_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None
    try:
        SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

filtered_backend_samples: dict[str, List[str]] = {}
backend_score_frames: dict[str, pd.DataFrame] = {}
for label, smiles_list in backend_samples.items():
    sanitized = [sanitize_smiles(s) for s in smiles_list]
    sanitized = [s for s in sanitized if s]
    result, status = get_backend_spec_by_label(label)
    if result is None or not sanitized:
        filtered_backend_samples[label] = []
        continue
    spec, scorers = result
    env = build_environment_and_modifier(scorers, spec.thresholds)
    score_df = env.getScores(sanitized)
    score_df["SMILES"] = sanitized
    mask = (score_df["Valid"] == 1) & (score_df["Desired"] == 1)
    filtered_backend_samples[label] = score_df.loc[mask, "SMILES"].tolist()
    backend_score_frames[label] = score_df

aggregate_samples = filtered_backend_samples
aggregate_scores = backend_score_frames
{label: len(smiles) for label, smiles in aggregate_samples.items()}


# %%
filtered_summary = pd.DataFrame.from_dict(
    {label: {"kept": len(smiles), "original": len(backend_samples.get(label, []))}
     for label, smiles in aggregate_samples.items()},
    orient="index"
)
filtered_summary["retention"] = filtered_summary["kept"] / filtered_summary["original"].replace({0: np.nan})
filtered_summary.sort_values("retention", ascending=False)


# %% [markdown]
# The summary table reports how many sanitized, desired molecules remain per backend
# compared with the original sampled totals. Expect higher retention once RL models
# stabilize and produce more goal satisfying candidates.
# 

# %% [markdown]
# The counts above reflect molecules that survived RDKit sanitization *and* met each backend's desirability thresholds.
# 

# %% [markdown]
# ### Native Score Summary
# 
# Diagonal entries in the cross-backend table reflect "native" scoring (each generator evaluated by its own backend). They serve as a quick reference for how consistently each policy attains its respective ROCS objective.
# 
# This section prepares filtered molecular samples for cross-backend evaluation by organizing sanitized, desired compounds from each backend into standardized collections. The native scores establish baseline performance metrics for each backend's trained policy, enabling systematic comparison of molecular generation quality across different conformer/scorer combinations.

# %% [markdown]
# ## Cross-Backend Scoring (filtered sets)
# 
# Each filtered, desired collection is rescored against every backend. Scores reflect freshly generated conformers, so sample counts in the table correspond to the post-filtered molecules above.
# 
# Systematic cross-validation quantifies generation consistency across conformer/scorer combinations, identifying robust molecular generation strategies for CCR ligands.

# %%
ALL_BACKEND_LABELS = [
    "OpenEye (Omega + ROCS)",
    "RDKit Aggregate",
    "RDKit Supermolecule",
    "CDPKit Aggregate",
    "CDPKit Supermolecule",
]

# Filtered (sanitized + desired) samples used downstream
aggregate_samples = {
    label: filtered_backend_samples.get(label, [])
    for label in ALL_BACKEND_LABELS
}

# Raw sampling sizes retained for reference
RAW_SAMPLE_SIZES = {label: len(backend_samples.get(label, [])) for label in ALL_BACKEND_LABELS}
FILTERED_SAMPLE_SIZES = {label: len(smiles) for label, smiles in aggregate_samples.items()}

cross_records = []
for source_label, smiles_list in aggregate_samples.items():
    if not smiles_list:
        continue
    for scorer_label in ALL_BACKEND_LABELS:
        try:
            scores = score_with_backend(scorer_label, smiles_list)
            status = "ok"
        except Exception as exc:
            scores = pd.Series(dtype=float)
            status = f"error: {exc}"
        cross_records.append({
            "sampled_from": source_label,
            "scored_by": scorer_label,
            "mean_score": float(np.nanmean(scores)) if len(scores) else np.nan,
            "std_score": float(np.nanstd(scores)) if len(scores) else np.nan,
            "n": int(len(scores)),
            "status": status,
        })
cross_score_df = pd.DataFrame(cross_records)
cross_score_df

# %% [markdown]
# ### Cross-Backend Analysis Results
# 
# Previous analysis will reveal which backend combinations produce the most consistent and high-scoring molecular sets, enabling identification of optimal generation strategies.

# %% [markdown]
# ## Model Comparison
# 
# With the RL agents already trained for OpenEye ROCS, RDKit, and CDPKit, we can sample fresh molecules from each policy and study how their chemotypes diverge.
# 
# This section compares molecular generation across all five backend configurations (OpenEye, RDKit Aggregate, RDKit Supermolecule, CDPKit Aggregate, CDPKit Supermolecule) to analyze chemotype diversity and generation consistency.

# %%
ALL_BACKEND_LABELS = [
    "OpenEye (Omega + ROCS)",
    "RDKit Aggregate",
    "RDKit Supermolecule",
    "CDPKit Aggregate",
    "CDPKit Supermolecule",
]

aggregate_samples = {
    label: backend_samples.get(label, [])
    for label in ALL_BACKEND_LABELS
    if backend_samples.get(label)
}

if len(aggregate_samples) != len(ALL_BACKEND_LABELS):
    missing = sorted(set(ALL_BACKEND_LABELS) - set(aggregate_samples))
    print("Warning: missing samples for " + ", ".join(missing))
    
sample_summary = (
    pd.Series({label: len(smiles) for label, smiles in aggregate_samples.items()}, name="samples")
    .rename_axis("backend")
    .to_frame()
    .sort_values("samples", ascending=False)
)

sample_summary

# %% [markdown]
# ### Murcko Scaffold Analysis
# 
# We map each generated SMILES to its Bemis-Murcko scaffold to highlight the structural families emphasized by every backend.
# 
# This analysis quantifies scaffold diversity across backend configurations, revealing structural bias patterns in molecular generation. Bemis-Murcko scaffolds capture core ring systems and linkers, providing insight into chemotype distribution and generation consistency.

# %%
from collections import Counter
from rdkit.Chem.Scaffolds import MurckoScaffold

def to_murcko_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaffold, canonical=True)


scaffold_counters: dict[str, Counter[str]] = {}
scaffold_records = []
for label, smiles_list in aggregate_samples.items():
    scaffolds = [to_murcko_scaffold(s) for s in smiles_list]
    scaffolds = [s for s in scaffolds if s]
    counter = Counter(scaffolds)
    scaffold_counters[label] = counter
    top = counter.most_common(1)
    scaffold_records.append(
        {
            "backend": label,
            "unique_scaffolds": len(counter),
            "top_scaffold": top[0][0] if top else None,
            "top_frequency": top[0][1] if top else 0,
        }
    )

scaffold_summary = pd.DataFrame(scaffold_records).sort_values(
    "unique_scaffolds", ascending=False
)
scaffold_summary


# %% [markdown]
# ### Scaffold Gallery
# 
# Visual inspection of scaffold patterns reveals both generation bias (top scaffolds) and structural diversity (diverse scaffolds) across backend configurations.
# 
# **Analysis Components:**
# - **Top scaffolds**: Shows generation bias and consistent structural preferences
# - **Diverse scaffolds**: Displays scaffolds from different frequency bins to assess exploration
# - **Overlap matrix**: Reveals structural diversity and convergence patterns between backends

# %%
# Top scaffolds (generation bias analysis)
TOP_K = 5
top_rows = []
for label, counter in scaffold_counters.items():
    for rank, (scaffold, count) in enumerate(counter.most_common(TOP_K), start=1):
        top_rows.append(
            {
                "backend": label,
                "rank": rank,
                "scaffold_smiles": scaffold,
                "count": count,
                "fraction": count / sample_summary.loc[label, "samples"],
            }
        )

top_scaffolds = pd.DataFrame(top_rows)
print("Top Scaffolds (Generation Bias):")
display(top_scaffolds)

# %%
# Diverse scaffolds (exploration analysis)
def get_diverse_scaffolds(counter, n_scaffolds=5):
    """Select scaffolds from different frequency bins for diversity analysis."""
    if not counter:
        return []
    
    # Sort scaffolds by frequency
    sorted_scaffolds = counter.most_common()
    total_scaffolds = len(sorted_scaffolds)
    
    if total_scaffolds <= n_scaffolds:
        return sorted_scaffolds
    
    # Select scaffolds from different frequency bins
    diverse_scaffolds = []
    bin_size = total_scaffolds // n_scaffolds
    
    for i in range(n_scaffolds):
        idx = min(i * bin_size, total_scaffolds - 1)
        diverse_scaffolds.append(sorted_scaffolds[idx])
    
    return diverse_scaffolds

diverse_rows = []
for label, counter in scaffold_counters.items():
    diverse_scaffolds = get_diverse_scaffolds(counter, n_scaffolds=5)
    for rank, (scaffold, count) in enumerate(diverse_scaffolds, start=1):
        diverse_rows.append(
            {
                "backend": label,
                "rank": rank,
                "scaffold_smiles": scaffold,
                "count": count,
                "fraction": count / sample_summary.loc[label, "samples"],
            }
        )

diverse_scaffolds_df = pd.DataFrame(diverse_rows)
print("\nDiverse Scaffolds (Exploration Analysis):")
display(diverse_scaffolds_df)

# %%
# Scaffold overlap analysis
scaffold_sets = {label: set(counter) for label, counter in scaffold_counters.items()}
overlap_matrix = pd.DataFrame(
    index=ALL_BACKEND_LABELS,
    columns=ALL_BACKEND_LABELS,
    dtype=int,
)
for a in ALL_BACKEND_LABELS:
    for b in ALL_BACKEND_LABELS:
        set_a = scaffold_sets.get(a, set())
        set_b = scaffold_sets.get(b, set())
        overlap_matrix.loc[a, b] = len(set_a & set_b)

print("\nScaffold Overlap Matrix:")
display(overlap_matrix)

# %%

# Improved Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_text_color(value, max_value):
    """Determine text color based on background intensity"""
    # Use white text for values above 60% of max, black otherwise
    threshold = 0.6 * max_value
    return 'white' if value > threshold else 'black'

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Improved heatmap
im = axes[0].imshow(overlap_matrix.values, cmap="Blues", aspect='auto')
axes[0].set_xticks(range(len(ALL_BACKEND_LABELS)))
axes[0].set_xticklabels(ALL_BACKEND_LABELS, rotation=45, ha="right", fontsize=10)
axes[0].set_yticks(range(len(ALL_BACKEND_LABELS)))
axes[0].set_yticklabels(ALL_BACKEND_LABELS, fontsize=10)
axes[0].set_title("Shared Murcko scaffolds", fontsize=12, fontweight='bold', pad=20)

# Add text with dynamic color
max_val = overlap_matrix.values.max()
for i in range(len(ALL_BACKEND_LABELS)):
    for j in range(len(ALL_BACKEND_LABELS)):
        value = overlap_matrix.values[i, j]
        text_color = get_text_color(value, max_val)
        axes[0].text(
            j, i, f"{value:.0f}",
            ha="center", va="center",
            color=text_color, fontweight='bold', fontsize=9
        )

# Add colorbar
cbar = plt.colorbar(im, ax=axes[0], shrink=0.8)
cbar.set_label('Number of shared scaffolds', rotation=270, labelpad=15)

# Improved bar chart
colors = plt.get_cmap("tab10").colors
bars = axes[1].bar(
    scaffold_summary["backend"],
    scaffold_summary["unique_scaffolds"],
    color=[colors[i] for i in range(len(scaffold_summary))],
    alpha=0.8,
    edgecolor='black',
    linewidth=0.5
)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')

axes[1].set_ylabel("Unique scaffolds", fontsize=11, fontweight='bold')
axes[1].set_title("Diversity per backend", fontsize=12, fontweight='bold', pad=20)
axes[1].tick_params(axis='x', labelsize=10)
axes[1].tick_params(axis='y', labelsize=10)
plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

# Add grid for better readability
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

# Improve overall layout
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.show()

# %% [markdown]
# ### Scaffold Concentration Metrics
# 
# Quantifies scaffold distribution inequality and generation concentration patterns across backends, providing context for novelty analysis against training data.
# 
# The table and bars highlight how much of each generator's output is concentrated in a handful of scaffold families versus long-tail exploration. High singleton ratios indicate the policy is still sampling many one-off chemotypes.

# %%
import numpy as np

def gini_coefficient(x):
    """Calculate Gini coefficient for distribution inequality.
    
    Args:
        x: Array of values (scaffold frequencies)
    
    Returns:
        float: Gini coefficient between 0 (equal) and 1 (maximum inequality)
    """
    # Ensure the array is sorted
    sorted_x = np.sort(x)
    n = len(x)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
    
    # Calculate cumulative sum
    cumulative_x = np.cumsum(sorted_x)
    # Calculate Gini coefficient
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_x))) / (n * np.sum(sorted_x)) - (n + 1) / n
    return gini

# Focus on genuinely new metrics
scaffold_metric_rows = []
for label, counter in scaffold_counters.items():
    total = int(sample_summary.loc[label, "samples"]) if label in sample_summary.index else len(aggregate_samples.get(label, []))
    unique = len(counter)
    
    # Gini coefficient for scaffold distribution inequality (corrected)
    frequencies = list(counter.values())
    gini = gini_coefficient(frequencies)
    
    # Top-5 concentration
    top5_count = sum(freq for _, freq in counter.most_common(5))
    
    scaffold_metric_rows.append({
        "backend": label,
        "unique_scaffolds": unique,
        "unique_fraction": unique / total if total else np.nan,
        "top5_fraction": top5_count / total if total else np.nan,
        "gini_coefficient": gini,  # 0 = equal distribution, 1 = maximum inequality
    })

scaffold_metric_df = (
    pd.DataFrame(scaffold_metric_rows)
    .sort_values("gini_coefficient", ascending=False)  # Sort by inequality
)
scaffold_metric_df

# %%
# Simplified visualization focusing on key insights
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.bar(scaffold_metric_df["backend"], scaffold_metric_df["gini_coefficient"], color="#2ca02c")
ax.set_ylabel("Gini coefficient")
ax.set_title("Scaffold distribution inequality (0=equal, 1=maximum concentration)")
ax.set_xticklabels(scaffold_metric_df["backend"], rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Novelty Against Transfer Set
# 
# Assesses molecular novelty by computing Morgan fingerprint similarities between generated molecules and the CCR training set. Lower maximum similarities indicate exploration beyond the transfer learning corpus.
# 
# This analysis quantifies the exploration-exploitation trade-off: high similarity suggests the model exploits learned patterns, while low similarity indicates exploration of novel chemical space.

# %%
### Molecular Similarity Analysis

from rdkit.Chem import AllChem

def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

transfer_smiles = [s for s in ccr_df["SMILES"].dropna().tolist()]
transfer_fps = [fp for smi in transfer_smiles if (fp := morgan_fp(smi)) is not None]

novelty_distributions: dict[str, pd.Series] = {}
novelty_rows = []
for label, smiles_list in aggregate_samples.items():
    similarities = []
    for smi in smiles_list:
        fp = morgan_fp(smi)
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, transfer_fps)
        similarities.append(max(sims) if sims else float("nan"))
    series = pd.Series(similarities, name=label).dropna()
    novelty_distributions[label] = series
    if series.empty:
        novelty_rows.append({
            "backend": label,
            "mean_max_similarity": np.nan,
            "median_max_similarity": np.nan,
            "fraction_below_0_4": np.nan,
            "fraction_below_0_6": np.nan,
        })
    else:
        novelty_rows.append({
            "backend": label,
            "mean_max_similarity": series.mean(),
            "median_max_similarity": series.median(),
            "fraction_below_0_4": (series < 0.4).mean(),
            "fraction_below_0_6": (series < 0.6).mean(),
        })

novelty_summary = pd.DataFrame(novelty_rows).sort_values("mean_max_similarity")
novelty_summary

# %% [markdown]
# ### Similarity Distribution Analysis
# 
# The distributions reveal the balance between exploitation (high similarity to training data) and exploration (low similarity, novel chemistry). Backends with broader distributions demonstrate more diverse generation strategies.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for label, series in novelty_distributions.items():
    clean = series.dropna()
    if len(clean) < 2:
        continue
    sns.kdeplot(clean, ax=axes[0], label=f"{label} (n={len(clean)})", bw_adjust=0.7, clip=(0, 1))
axes[0].set_xlabel("Max Tanimoto similarity to transfer set")
axes[0].set_ylabel("Density")
axes[0].set_title("Similarity distributions")
axes[0].legend()

for label, series in novelty_distributions.items():
    clean = series.dropna()
    if len(clean) < 2:
        continue
    novelty = 1 - clean
    sns.kdeplot(novelty, ax=axes[1], label=f"{label} (n={len(clean)})", bw_adjust=0.7, clip=(0, 1))
axes[1].set_xlabel("Novelty (1 - max similarity)")
axes[1].set_ylabel("Density")
axes[1].set_title("Novelty distributions")
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Novelty Quantiles
# 
# Quantile analysis reveals the distribution of novelty across generated molecules, identifying which backends produce more consistently novel or more consistently similar molecules to the training set.
# 
# The novelty plots diagnose whether any backend collapses back onto training chemistry (curves skewed toward similarity 1.0) or continues to explore unseen regions (mass near novelty 1.0).

# %%
novelty_stats = []
for label, series in novelty_distributions.items():
    clean = series.dropna()
    if clean.empty:
        continue
    novelty = 1 - clean
    novelty_stats.append({
        "backend": label,
        "similarity_p25": clean.quantile(0.25),
        "similarity_p50": clean.quantile(0.50),
        "similarity_p75": clean.quantile(0.75),
        "novelty_p25": novelty.quantile(0.25),
        "novelty_p50": novelty.quantile(0.50),
        "novelty_p75": novelty.quantile(0.75),
    })
novelty_percentiles = pd.DataFrame(novelty_stats)
novelty_percentiles

# %% [markdown]
# ### Scaffold-Level Novelty
# 
# Complement molecular similarity with scaffold-level novelty by comparing Bemis-Murcko scaffolds between generated and training sets.

# %%
# Scaffold-level novelty analysis
transfer_scaffolds = set()
for smi in transfer_smiles:
    scaffold = to_murcko_scaffold(smi)
    if scaffold:
        transfer_scaffolds.add(scaffold)

scaffold_novelty_rows = []
for label, counter in scaffold_counters.items():
    generated_scaffolds = set(counter.keys())
    novel_scaffolds = generated_scaffolds - transfer_scaffolds
    shared_scaffolds = generated_scaffolds & transfer_scaffolds
    
    scaffold_novelty_rows.append({
        "backend": label,
        "total_scaffolds": len(generated_scaffolds),
        "novel_scaffolds": len(novel_scaffolds),
        "shared_scaffolds": len(shared_scaffolds),
        "novelty_fraction": len(novel_scaffolds) / len(generated_scaffolds) if generated_scaffolds else 0,
    })

scaffold_novelty_df = pd.DataFrame(scaffold_novelty_rows).sort_values("novelty_fraction", ascending=False)
scaffold_novelty_df

# %% [markdown]
# ## Native Score Distributions
# 
# Analyzes score distribution shapes for each backend using native scoring pipelines. Distribution analysis reveals generation consistency: narrow distributions indicate consistent high-quality generation, while broad or multimodal distributions suggest diverse but potentially inconsistent output.

# %%
# Native scoring with each backend's own toolkit
native_scores: dict[str, pd.DataFrame] = {}
for label in ALL_BACKEND_LABELS:
    smiles_list = aggregate_samples.get(label, [])
    if not smiles_list:
        continue
    result, status = get_backend_spec_by_label(label)
    if result is None:
        continue
    spec, scorers = result
    env = build_environment_and_modifier(scorers, spec.thresholds)
    df_scores = env.getScores(smiles_list)
    df_scores["SMILES"] = smiles_list
    native_scores[label] = df_scores

print(f"Native scoring completed for: {list(native_scores.keys())}")

# %% [markdown]
# ### Distribution Visualization
# 
# Histograms reveal score distribution shapes, diagnosing whether backends produce consistently high scores (narrow, high-centered distributions) or explore diverse chemical space with variable scoring (broad distributions).

# %%
# Extract scoring metrics for visualization
score_metrics = set()
for df_scores in native_scores.values():
    numeric_cols = df_scores.select_dtypes(include=[float, int]).columns
    score_metrics.update(numeric_cols)

# Filter out non-score columns
metrics_to_plot = sorted([m for m in score_metrics if m not in {"Valid", "Desired", "SMILES"}])

if metrics_to_plot:
    palette = dict(zip(ALL_BACKEND_LABELS, sns.color_palette("colorblind", len(ALL_BACKEND_LABELS))))
    bins = np.linspace(0, 1, 21)
    
    # Calculate grid dimensions: 2 plots per row
    n_metrics = len(metrics_to_plot)
    n_rows = (n_metrics + 1) // 2  # Ceiling division
    n_cols = min(2, n_metrics)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
    
    # Handle single subplot case
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if n_metrics > 1 else axes
    
    for i, metric in enumerate(metrics_to_plot):
        axis = axes_flat[i]
        
        for label, df_scores in native_scores.items():
            if metric not in df_scores.columns:
                continue
            clean = df_scores[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean) < 2:
                continue
            axis.hist(
                clean,
                bins=bins,
                density=True,
                histtype='step',
                linewidth=2,
                color=palette[label],
                label=f"{label} (n={len(clean)})"
            )
        axis.set_title(f"{metric} distribution")
        axis.set_xlabel("Score")
        axis.grid(alpha=0.3)
    
    # Set ylabel only for leftmost plots
    for i in range(0, n_metrics, 2):
        axes_flat[i].set_ylabel("Density")
    
    # Add legend to the last subplot
    if n_metrics > 0:
        axes_flat[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
else:
    print("No numeric metrics available for distribution plots.")

# %% [markdown]
# ## Reinforcement Learning Progress
# 
# Analyzes RL training dynamics to validate convergence and contextualize molecular generation quality. Training metrics reveal whether backends successfully optimize for ROCS objectives.

# %%
# Load RL training monitor logs
rl_metric_frames = []
for label, spec in BACKENDS.items():
    fit_path = spec.rl_dir / f"{spec.monitor_prefix}_fit.tsv"
    if fit_path.exists():
        df = pd.read_csv(fit_path, sep="\t")
        df["backend"] = label
        rl_metric_frames.append(df)


# %% [markdown]
# ### Training Dynamics
# 
# Plots reveal convergence patterns and training stability across backends. Plateau behavior indicates optimization trade-offs between objectives.

# %%
if not rl_metrics.empty:
    metrics_to_plot = ["desired_ratio", "unique_ratio", "valid_ratio", "avg_amean"]
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 12), sharex=True)
    
    for metric, ax in zip(metrics_to_plot, axes):
        for label, group in rl_metrics.groupby("backend"):
            ax.plot(group["Epoch"], group[metric], marker="o", markersize=3, label=label, linewidth=1.5)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("RL Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()
    
    # Endpoint summary
    latest = (rl_metrics.sort_values(["backend", "Epoch"])
              .groupby("backend")
              .tail(1)
              .set_index("backend"))
    rl_endpoint_summary = latest[metrics_to_plot].round(3)
    print("\nFinal Training Metrics:")
    display(rl_endpoint_summary)
else:
    print("No RL metrics available. Training has not been completed yet.")


