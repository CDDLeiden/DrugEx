# DrugEx ROCS Tutorial: Shape-Based Molecular Generation

This advanced tutorial demonstrates how to use DrugEx with ROCS (Rapid Overlay of Chemical Structures) shape-based scoring for de novo drug design. We apply reinforcement learning to generate novel CCR2 ligand candidates optimized for 3D shape similarity to known actives.

## Overview

Shape-based molecular design uses 3D similarity to known active compounds as a guide for generating new molecules. This tutorial compares three ROCS implementations:

- **RDKit ROCS** - Open-source, Python-native
- **CDPKit ROCS** - Open-source, advanced features
- **OpenEye ROCS** - Commercial, industry standard

## Prerequisites

### Background Knowledge

**Important**: This is an advanced tutorial. We recommend completing the basic DrugEx tutorials first:

1. **`tutorial/pretrain.ipynb`** - Understanding pretrained language models for chemistry
2. **`tutorial/finetune.ipynb`** - Transfer learning and domain adaptation
3. **`tutorial/rl_*.ipynb`** - Introduction to reinforcement learning with DrugEx

These tutorials cover essential concepts like:
- SMILES encoding and vocabulary
- Transfer learning from pretrained models
- Reinforcement learning basics
- Multi-objective optimization with Pareto ranking

### Software Requirements

**Required**:
- Python 3.8+
- DrugEx (with RDKit backend)
- PyTorch
- Pandas, NumPy

**Optional** (for comparing backends):
- CDPKit: `pip install drugex[cdpkit]`
- OpenEye toolkit (requires license)

## Quick Start: 3-Step Workflow

### Step 1: Fine-Tune Model
```bash
python prepare_models.py
```
Fine-tunes a pretrained model on CCR2 ligand data (~10-20 minutes).

### Step 2: Train RL Model
```bash
jupyter notebook quickstart_rl_rdkit.ipynb
```
Minimal notebook showing just the RL training loop (~30-60 minutes for 30 epochs).

### Step 3: Generate Molecules
```bash
python generate_molecules.py
```
Generates novel molecules from the trained model.

That's it! Simple and focused.

## Tutorial Structure

### Core Files

| File | Purpose | Type | Required |
|------|---------|------|----------|
| `config.py` | Predefined settings & setup helper | Module | Yes |
| `prepare_models.py` | Fine-tune model on CCR2 data | Script | **Yes** |
| `quickstart_rl_rdkit.ipynb` | Minimal RL training notebook | Notebook | **Yes** |
| `generate_molecules.py` | Generate molecules from trained model | Script | **Yes** |
| `run_cdpkit_rocs.py` | RL with CDPKit ROCS backend | Script | Optional |
| `run_openeye_rocs.py` | RL with OpenEye ROCS backend | Script | Optional |
| `rocs_rl_ccr/` | CCR2 reference ligands and data | Data | - |

### Reference Files

| File | Purpose |
|------|---------|
| `ROCS-RNN-Demo_v2_new.ipynb` | Comprehensive demo (kept as reference) |
| `demo_out/` | Output directory for models and datasets |
| `rl_runs_demo/` | RL training results and generated molecules |

## Detailed Guide

### 1. Configuration (`config.py`)

All settings are predefined in `config.py`:

```python
from config import setup_rl_rdkit, get_config

# Get all configuration parameters
config = get_config()
# Returns: RL_EPOCHS=30, RL_EPSILON=0.2, RL_N_SAMPLES=500, etc.

# One-line setup for RL experiment
agent, mutate, env, output_dir = setup_rl_rdkit()
```

**To customize**: Edit `config.py` to change:
- RL parameters (epochs, epsilon, samples)
- Conformer settings (max_conformers, max_isomers)
- Reference ligands path
- Output directories

### 2. Model Preparation (`prepare_models.py`)

Fine-tunes a pretrained DrugEx model on CCR2-specific data.

```bash
# Run with defaults
python prepare_models.py

# Customize training
python prepare_models.py --epochs 50 --batch-size 128
```

**Options**:
- `--epochs`: Number of fine-tuning epochs (default: 100)
- `--batch-size`: Training batch size (default: 256)
- `--force`: Force retraining even if model exists

**Output**:
- `demo_out/models/CCR2_finetuned.pkg` - Fine-tuned model
- `demo_out/models/CCR2_finetuned.vocab` - Vocabulary
- Training metrics in TSV files

### 3. RL Training (`quickstart_rl_rdkit.ipynb`)

**Ultra-minimal notebook** with just 6 cells:

1. **Imports** - Load necessary modules
2. **Setup** - One-line setup from config
3. **Create Explorer** - RL explorer configuration
4. **Train** - Run RL training loop
5. **Plot Results** - Visualize training progress
6. **Next Steps** - Links to generation and other backends

**No configuration needed** - all settings come from `config.py`.

### 4. Molecule Generation (`generate_molecules.py`)

Generates novel molecules from the trained RL model.

```bash
# Generate with defaults (100 molecules)
python generate_molecules.py

# Customize generation
python generate_molecules.py --num-samples 500

# Use specific model
python generate_molecules.py --model path/to/model.pkg --num-samples 200
```

**Options**:
- `--model`: Path to trained model (default: from config)
- `--num-samples`: Number of molecules to generate (default: 100)
- `--output`: Custom output file path

**Output**:
- TSV file with generated molecules and scores
- Console summary with statistics and top molecules

### 5. Alternative Backends (Optional)

After completing the quick start, try other ROCS implementations:

#### CDPKit ROCS

```bash
pip install drugex[cdpkit]
python run_cdpkit_rocs.py --epochs 30
```

**Features**:
- Multi-stereoisomer scoring (`max_isomers` parameter)
- Optimized conformer generation
- Open-source

#### OpenEye ROCS

```bash
python run_openeye_rocs.py --epochs 30 --use-gpu
```

**Features**:
- Industry-standard implementation
- GPU acceleration
- Advanced color force field
- Requires commercial license

## Customization Guide

### Change RL Parameters

Edit `config.py`:

```python
# More thorough training
RL_EPOCHS = 50
RL_EPSILON = 0.3
RL_N_SAMPLES = 1000

# Faster training (for testing)
RL_EPOCHS = 10
RL_N_SAMPLES = 250
```

### Use Your Own Reference Ligands

1. Place your SDF file in the tutorial directory
2. Edit `config.py`:
```python
CCR2_SDF = ROOT / 'your_references.sdf'
```

### Adjust Conformer Generation

Edit `config.py`:

```python
# More conformers = better shape matching, slower
MAX_CONFORMERS = 50

# Fewer conformers = faster, less accurate
MAX_CONFORMERS = 10

# Consider more stereoisomers
MAX_ISOMERS = 4
```

### Add More Scoring Functions

Edit `config.py`, modify `create_rdkit_environment()`:

```python
from drugex.training.scorers.properties import Property

def create_rdkit_environment():
    rocs_scorer = RDKitROCSScorer(...)
    sa_scorer = Property('SA')
    sa_scorer.setModifier(SmoothClippedScore(lower_x=5, upper_x=3))

    # Add QED (drug-likeness)
    qed_scorer = Property('QED')

    env = DrugExEnvironment(
        scorers=[rocs_scorer, sa_scorer, qed_scorer],
        reward_scheme=ParetoCrowdingDistance()
    )
    return env
```

## Performance Tips

### Speed vs. Accuracy

**For faster experimentation**:
- Reduce `RL_EPOCHS` (e.g., 15-20)
- Reduce `MAX_CONFORMERS` (e.g., 10-15)
- Reduce `RL_N_SAMPLES` (e.g., 250)

**For better results**:
- Increase `RL_EPOCHS` (e.g., 50-100)
- Increase `MAX_CONFORMERS` (e.g., 50)
- Enable more isomers (CDPKit only)

### Computational Resources

**CPU**:
- All scorers use `n_jobs=-1` (all cores)
- 20-40 cores recommended
- Memory scales with batch size

**GPU**:
- PyTorch automatically uses GPU for NN training
- OpenEye Omega supports GPU (`use_gpu=True`)

## Troubleshooting

### Common Issues

**"Fine-tuned model not found"**
```bash
python prepare_models.py
```

**"CDPKit not available"**
```bash
pip install drugex[cdpkit]
```

**"OpenEye ROCS not found"**
- Install OpenEye toolkit
- Configure license
- Ensure `rocs` is in PATH

**Slow performance**
- Reduce `MAX_CONFORMERS`
- Reduce `RL_N_SAMPLES`
- Check CPU/GPU utilization

## Understanding Results

### Training Metrics

In the notebook plot and `*_fit.tsv` files:

- **desired_ratio**: Fraction of molecules meeting objectives (higher = better)
- **avg_amean**: Average score across objectives (higher = better)

### Generated Molecules

In `*_generated.tsv`:

- **SMILES**: Generated molecule structures
- **Score columns**: Individual objective scores (0-1 scale)
- **Total**: Average across objectives
- **Desired**: Boolean flag (meets all objectives)

## Workflow Comparison

### New Simplified Workflow

```
prepare_models.py → quickstart_rl_rdkit.ipynb → generate_molecules.py
     (10 min)              (30-60 min)                  (instant)
```

**Benefits**:
- ✓ Minimal, focused notebook
- ✓ Configuration separated
- ✓ Generation separated
- ✓ Easy to customize
- ✓ Clear 3-step process

### Old Workflow

```
One large notebook with everything
```

**Issues**:
- ✗ Complex and overwhelming
- ✗ Configuration mixed with code
- ✗ Harder to customize
- ✗ Difficult to navigate

## Advanced Usage

### Batch Generation

Generate multiple batches:

```bash
for i in {1..5}; do
    python generate_molecules.py --num-samples 100 --output batch_${i}.tsv
done
```

### Custom Evaluation

```python
from config import create_rdkit_environment

env = create_rdkit_environment()
scores = env.getScores(smiles_list)
```

### Hyperparameter Tuning

Edit `config.py` and run multiple experiments:

```bash
# config.py: RL_EPOCHS = 20, RL_EPSILON = 0.1
jupyter notebook quickstart_rl_rdkit.ipynb

# config.py: RL_EPOCHS = 20, RL_EPSILON = 0.3
jupyter notebook quickstart_rl_rdkit.ipynb

# Compare results
```

## Citation

If you use this workflow in your research, please cite:

- DrugEx: [citation needed]
- ROCS methodology papers

## Support

For questions and issues:

- DrugEx documentation: [link]
- GitHub issues: [link]
- Discussions: [link]

## Additional Resources

- [RDKit documentation](https://www.rdkit.org/docs/)
- [CDPKit documentation](https://cdpkit.org/)
- [OpenEye ROCS](https://docs.eyesopen.com/applications/rocs/)
- [DrugEx paper](https://arxiv.org/abs/...)

---

**Remember**: This is an advanced tutorial. If you're new to DrugEx, start with the basic tutorials in `tutorial/` first!
