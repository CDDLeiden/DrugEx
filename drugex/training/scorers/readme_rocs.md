# DrugEx-ROCS Scorers

FastROCS-based scorers for molecular shape similarity scoring in DrugEx-ROCS.

## Available Scorers

### 1. OpenEyeScorer (`fastrocs.py`)

High-performance implementation for large-scale screening and production environments.

**Key Features:**
- Optimized for performance with advanced memory management
- Multi-threading and GPU acceleration
- Resource-aware batch sizing and efficient caching

**Configuration:**
```python
OpenEyeScorer(
    sq_model_path,              # Path to ROCS query file (.sq)
    use_gpu=True,               # Use GPU acceleration if available
    max_isomers=4,              # Maximum isomers per molecule
    max_rot_bonds=10,           # Maximum rotatable bonds threshold
    max_heavy_atoms=30,         # Maximum heavy atoms threshold
    max_conformers=10,          # Maximum conformers per molecule
    cpu_processes=None          # CPU processes (auto-determined if None)
)
```

### 2. RocsScorer (`ez_rocs.py`)

Simplified implementation with file-based workflow for development and educational use.

**Key Features:**
- Easy to understand with minimal dependencies
- Compatible with both CPU and GPU modes
- Configurable score type extraction (TanimotoCombo, ShapeTanimoto, ColorTanimoto)

**Configuration:**
```python
RocsScorer(
    sq_model_path=None,         # Path to ROCS query file
    query_file=None,            # Alternative path to query file
    experiment_name="rocs_experiment", # Name for file naming
    score_type="TanimotoCombo", # Score type to extract
    use_gpu=False,              # Use GPU acceleration
    max_isomers=4,              # Maximum isomers per molecule
    max_rot_bonds=15,           # Maximum rotatable bonds threshold
    max_heavy_atoms=45,         # Maximum heavy atoms threshold
    max_conformers=10,          # Maximum conformers per molecule
    cpu_processes=None          # CPU processes (auto-determined if None)
)
```

## Choosing Between Scorers

- **OpenEyeScorer**: For production and high-volume screening. Features resource optimization and handles thousands of molecules efficiently.

- **RocsScorer**: For development, educational purposes, or modest molecule sets. Has configurable score types and clearer workflow.

## Key Technical Differences

### Default Filtering Thresholds
- **OpenEyeScorer**: 10 rotatable bonds, 30 heavy atoms (stricter)
- **RocsScorer**: 15 rotatable bonds, 45 heavy atoms (more permissive)

### Memory Management
- **OpenEyeScorer**: Adaptive resource allocation with sophisticated caching
- **RocsScorer**: Fixed thread allocation with simpler memory management

### GPU Acceleration
- **OpenEyeScorer**: Advanced GPU optimization with memory cleanup
- **RocsScorer**: Basic GPU support via standard configuration

## Usage Examples

### Basic Usage with OpenEyeScorer

```python
from drugex.training.scorers.fastrocs import OpenEyeScorer

scorer = OpenEyeScorer(
    sq_model_path="path/to/query.sq",
    use_gpu=True
)

scores = scorer(molecules_list)  # Can call directly as a function
# or
scores = scorer.getScores(smiles_list)
```

### Basic Usage with RocsScorer

```python
from drugex.training.scorers.ez_rocs import RocsScorer

scorer = RocsScorer(
    sq_model_path="path/to/query.sq",
    score_type="TanimotoCombo",
    use_gpu=False
)

scores = scorer.getScores(smiles_list)
```

### Using Different Score Types with RocsScorer

```python
# Extract shape-only similarity
shape_scorer = RocsScorer(
    sq_model_path="path/to/query.sq",
    score_type="ShapeTanimoto"
)

# Extract color (chemical features) similarity
color_scorer = RocsScorer(
    sq_model_path="path/to/query.sq",
    score_type="ColorTanimoto"
)

# Get different similarity types
shape_scores = shape_scorer.getScores(smiles_list)
color_scores = color_scorer.getScores(smiles_list)
```

### Processing Large Datasets with OpenEyeScorer

```python
scorer = OpenEyeScorer(
    sq_model_path="path/to/query.sq",
    use_gpu=True,
    max_isomers=2,              # Reduce isomer enumeration
    max_rot_bonds=7,            # More restrictive filtering
    max_heavy_atoms=25,         # More restrictive filtering
    max_conformers=5,           # Fewer conformers for faster processing
    cpu_processes=4             # Explicit process count
)

# Process in batches
batch_size = 1000
all_scores = []

for i in range(0, len(all_smiles), batch_size):
    batch = all_smiles[i:i+batch_size]
    scores = scorer.getScores(batch)
    all_scores.append(scores)
    
import numpy as np
final_scores = np.concatenate(all_scores)
```
