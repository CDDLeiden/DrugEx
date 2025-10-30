# DrugEx ROCS Tutorial: Shape-Based Molecular Generation

Train DrugEx with ROCS (Rapid Overlay of Chemical Structures) shape-based scoring for de novo drug design. Generate novel CCR2 ligand candidates optimized for 3D shape similarity to known actives.

## Prerequisites

**Background Knowledge:** Complete basic DrugEx tutorials first:
- `tutorial/Sequence-RNN.ipynb` - Introduction to molecular generation
- `tutorial/Graph-Transformer.ipynb` - Advanced model architectures
- Basic RL tutorials (`tutorial/rl_*.ipynb`) - Reinforcement learning concepts

## Quick Start

### Step 1: Fine-Tune Model

Fine-tune pretrained model on CCR2 ligands:

```bash
python prepare_models.py
```

**Options:**
- `--epochs 50` - Number of epochs (default: 100)
- `--batch-size 128` - Batch size (default: 256)
- `--force` - Force retrain

**Output:** `demo_out/models/CCR2_finetuned.pkg`

### Step 2: RL Training

Open the quickstart notebook:

```bash
jupyter notebook quickstart_rl_rdkit.ipynb
```

This notebook demonstrates the RL training loop with RDKit ROCS scoring. All settings are configured in `config.py` - no additional configuration needed.


### Step 3: Generate Molecules

Generate optimized molecules from the trained model:

```bash
# Generate 100 molecules (default)
python generate_molecules.py

# Custom amount
python generate_molecules.py --num-samples 500
```

**Output:** TSV file with SMILES, scores, and desired flags

## File Structure

| File | Purpose |
|------|---------|
| `config.py` | Configuration and environment setup |
| `prepare_models.py` | Fine-tune pretrained model on CCR2 data |
| `quickstart_rl_rdkit.ipynb` | **Main tutorial** - RL training with RDKit ROCS |
| `generate_molecules.py` | Generate molecules from trained model |
| `run_cdpkit_rocs.py` | Alternative: CDPKit ROCS backend |
| `run_openeye_rocs.py` | Alternative: OpenEye ROCS backend |

## Alternative ROCS Backends

### CDPKit (Open Source)

```bash
python run_cdpkit_rocs.py --epochs 30
```

Features: Multi-stereoisomer scoring, optimized conformer generation

### OpenEye (Commercial)

```bash
python run_openeye_rocs.py --epochs 30 --use-gpu
```

Features: GPU acceleration, industry-standard implementation (requires license)

## Configuration

Edit `config.py` to customize training parameters:

```python
# RL parameters
RL_EPOCHS = 50          # Training epochs
RL_EPSILON = 0.2        # Exploration rate
RL_N_SAMPLES = 1000     # Molecules per epoch

# Conformer generation
MAX_CONFORMERS = 50     # Conformers per molecule
MAX_ISOMERS = 4         # Stereoisomers to enumerate

# Scoring thresholds
ROCS_THRESHOLD = 0.871  # ROCS TanimotoCombo threshold (0-2 range)
SA_THRESHOLD = 0.1      # Synthetic accessibility threshold (0-1 range)
```

## Troubleshooting

**"Fine-tuned model not found"**: Run `python prepare_models.py` first

**Slow performance**: Reduce `MAX_CONFORMERS` and `RL_N_SAMPLES` in `config.py`

**CDPKit not available**: Install with `pip install cdpkit`

**Model training errors**: Ensure you've completed basic tutorials and understand DrugEx workflow

---

**New to DrugEx?** Start with basic tutorials in `tutorial/` first!
