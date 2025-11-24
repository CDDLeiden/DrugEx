# DrugEx ROCS Tutorial: Shape-Based Molecular Generation

[← Tutorial Index](../../README.md)

Reinforcement learning tutorial for generating CCR2 ligands optimized for ROCS shape similarity and synthetic accessibility using RDKit.

## Prerequisites

**Required:**
- RDKit (shape-based scoring backend)

**Optional:**
- CDPKit - Open-source alternative for higher accuracy
- OpenEye ROCS/VROCS Python toolkit - Commercial implementation with GPU support

## Reference Ligand Input Formats

ROCS scorers accept reference ligands in multiple formats:

| Format | Extension | Description | Backend Support |
|--------|-----------|-------------|-----------------|
| **SDF** | `.sdf` | Structure-Data File (standard) | All backends |
| **MOL2** | `.mol2` | Tripos MOL2 format | All backends |
| **Shape Query** | `.sq` | OpenEye VROCS Shape Query | OpenEye only |

**Requirements:**
- Must contain 3D coordinates (not 2D structures)
- Energy-minimized geometry recommended
- Appropriate protonation state

**Shape Query (.sq) Files:**
- Created by OpenEye VROCS GUI tool
- Pre-computed shape/pharmacophore features
- Faster scoring than on-the-fly SDF conversion
- Not portable to RDKit/CDPKit backends

**Examples:**
```python
# Single reference file
scorer = RDKitROCSScorer(references="reference.sdf")

# Multiple reference files
scorer = RDKitROCSScorer(references=["ref1.sdf", "ref2.sdf"])

# Shape Query with OpenEye backend
scorer = OpenEyeROCSScorer(references="reference.sq")

# Grouped scoring (multiple binding modes)
scorer = CDPKitROCSScorer(
    group_definitions=[
        ("active_site", ["active1.sdf"]),
        ("allosteric", ["allosteric1.sdf"])
    ]
)
```

## Quick Start

### 1. Fine-Tune Model
```bash
python prepare_models.py
```
**Required first step.** Fine-tunes pretrained model on CCR2 ligand data (default: 100 epochs).

### 2. Run Tutorial Notebook
```bash
jupyter notebook rocs_rl_tutorial.ipynb
```

### 3. Generate Molecules
```bash
python generate_molecules.py
```

## Notebook Workflow

The `rocs_rl_tutorial.ipynb` notebook walks through the complete RL training pipeline.

### Step 1: Threshold Validation

**Purpose:** Validate the ROCS threshold that determines which molecules receive rewards during RL training.

**What it does:**
- Compares 75 CCR2 active ligands vs 500 decoys
- Performs ROC analysis to find optimal threshold
- Uses Youden's Index to maximize active/decoy separation
- Validates if `config.py` threshold setting is appropriate

**Why it matters:**
- Threshold too high → model misses actual actives
- Threshold too low → model rewards decoy molecules
- Proper threshold ensures effective RL learning

**Output:** ROC curve plots and optimal threshold recommendation. If optimal differs significantly from config setting, adjust `ROCS_THRESHOLD` in `config.py`.

### Step 2: Setup

Loads pre-trained and fine-tuned models, creates RDKit ROCS environment with configured parameters:
- Pretrained agent (general chemical space)
- Fine-tuned mutate network (CCR2-specific)
- ROCS scorer (shape similarity to reference ligands)
- SA scorer (synthetic accessibility)

### Step 3: RL Explorer Configuration

Creates explorer with:
- `epsilon=0.2` - Exploration rate (20% random sampling)
- `n_samples=1000` - Molecules generated per epoch
- `epochs=50` - Training iterations

### Step 4: Training Loop

Runs RL training for configured epochs. Each epoch:
1. Generates molecules from current policy
2. Scores with ROCS + SA
3. Calculates rewards (Pareto crowding distance)
4. Updates model via policy gradient

Monitor: `desired_ratio` (fraction meeting thresholds) should increase over epochs.

### Step 5: Results Analysis

**Training metrics plot:**
- Desired molecules ratio (target: >0.5 by final epochs)
- Average score (should increase steadily)

**SA score evolution:**
- Tracks synthetic accessibility across training
- Should remain high (easier synthesis) while ROCS improves

**Output files:**
- `CCR2_rdkit_reinforced.pkg` - Trained model
- `CCR2_rdkit_reinforced_fit.tsv` - Training metrics
- `CCR2_rdkit_reinforced_smiles.tsv` - All generated molecules

## Understanding Threshold Analysis

The threshold analysis script (`threshold_analysis.py`) uses ROC methodology:

1. **Score all molecules** - Actives and decoys with same ROCS settings as RL
2. **Build ROC curve** - True positive rate vs false positive rate
3. **Find optimal threshold** - Maximize Youden's Index (TPR - FPR)
4. **Compare to config** - Check if current setting is near-optimal

**When to adjust threshold:**
- Optimal differs by >0.2 from config value
- Low desired_ratio during training (<0.3)
- ROC AUC < 0.8 (poor separation)

Edit `ROCS_THRESHOLD` in `config.py` based on analysis results.

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RL_EPOCHS` | 50 | Training epochs |
| `RL_EPSILON` | 0.2 | Exploration rate |
| `RL_N_SAMPLES` | 1000 | Molecules per epoch |
| `MAX_CONFORMERS` | 50 | Conformers per molecule |
| `MAX_ISOMERS` | 4 | Stereoisomers to enumerate |
| `ROCS_THRESHOLD` | 0.871 | ROCS TanimotoCombo cutoff |
| `SA_THRESHOLD` | 0.1 | Synthetic accessibility cutoff |

## File Structure

| File | Purpose |
|------|---------|
| `rocs_rl_tutorial.ipynb` | **Main tutorial** - Complete RL workflow |
| `config.py` | Configuration and setup functions |
| `prepare_models.py` | Fine-tune pretrained model on CCR2 data |
| `threshold_analysis.py` | ROC analysis for threshold optimization |
| `generate_molecules.py` | Generate molecules from trained model |
| `run_cdpkit_rocs.py` | Alternative: CDPKit backend |
| `run_openeye_rocs.py` | Alternative: OpenEye backend |

## Alternative ROCS Backends

**CDPKit (Open Source):** Open-source. Install: `pip install cdpkit`, then run `python run_cdpkit_rocs.py`.

**OpenEye (Commercial):** Requires valid license. Run `python run_openeye_rocs.py --use-gpu`.

## Troubleshooting

**Model not found:**
```bash
python prepare_models.py
```

**Slow training:**
- Reduce `MAX_CONFORMERS` in `config.py` (try 20-30)
- Reduce `RL_N_SAMPLES` (try 500)
- Use fewer RL epochs for testing

**Low desired_ratio:**
- Check threshold analysis results
- Adjust `ROCS_THRESHOLD` in `config.py`
- Verify reference ligands are appropriate

**CDPKit installation:**
```bash
conda install -c conda-forge cdpkit
```

**OpenEye license:**
```bash
export OE_LICENSE=/path/to/oe_license.txt
python -c "from openeye import oechem; print(oechem.OEChemIsLicensed())"
```
