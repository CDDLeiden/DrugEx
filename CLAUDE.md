# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation
```bash
# Install the package in development mode
pip install -e .

# Install with optional QSPRPred dependencies  
pip install -e ".[qsprpred]"

# Install development dependencies
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests with pytest
python -m pytest

# Run specific test file
python -m pytest drugex/training/tests.py

# Run tests in specific directory
python -m pytest drugex/data/
```

### Package Usage
```bash
# Download tutorial models and data
python -m drugex.download

# Use the drugex CLI script
./scripts/drugex <command>
```

## Architecture Overview

### Core Components

**drugex/** - Main package containing the core DrugEx functionality:
- **data/** - Dataset management, corpus handling, vocabulary, and molecular processing
- **training/** - Training infrastructure including generators, explorers, rewards, and scorers
- **molecules/** - Molecular representation and conversion utilities
- **parallel/** - Parallel processing and evaluation interfaces
- **utils/** - General utility functions and helper classes

**Key Training Components:**
- **generators/** - Neural network architectures (RNN, Transformers) for molecule generation
- **explorers/** - Exploration strategies for reinforcement learning
- **scorers/** - Molecular scoring functions including ROCS implementations
- **rewards/** - Reward calculation and multi-objective optimization

### ROCS Integration

The project includes extensive ROCS (Rapid Overlay of Chemical Structures) integration for shape-based molecular similarity scoring:

**drugex/training/scorers/** contains multiple ROCS implementations:

**Commercial OpenEye ROCS Implementations:**
- `base_rocs.py` - Base ROCS scorer implementations (CPU and GPU FastROCS)
- `ez_rocs.py` - Simplified ROCS scorer with minimal dependencies  
- `fastrocs.py` - Original FastROCS implementations
- `cli_rocs.py` - Command-line interface for ROCS scoring
- `smart_cli_rocs.py` - Memory-optimized CLI ROCS with OOM prevention

**Open-Source ROCS Alternatives:**
- `cdpkit_scorers.py` - CDPKit-based shape scoring (high accuracy, no license required)
- `rdkit_scorers.py` - RDKit-based molecular scoring (high speed, no license required)

**Supporting Infrastructure:**
- `conformer_generators.py` - Conformer generation for all ROCS variants (OpenEye, CDPKit, RDKit)

### ROCS Implementation Decision Matrix

| Implementation | License | Accuracy | Speed | Memory | GPU Support | Best For |
|----------------|---------|----------|-------|--------|-------------|----------|
| **OpenEye FastROCS** | Commercial | Highest | Fastest | Moderate | Yes | Production, High-throughput |
| **CDPKit Scorers** | Open Source | High | Moderate | Higher | No | Research, High accuracy |
| **RDKit Scorers** | Open Source | Good | Fast | Moderate | No | Development, Prototyping |

### ROCS Installation Requirements

**OpenEye ROCS (Commercial):**
```bash
# Requires valid OpenEye license
export OE_LICENSE=/path/to/oe_license.txt
conda install -c openeye openeye-toolkits
```

**CDPKit (Open Source):**
```bash
# High accuracy alternative
conda install -c conda-forge cdpkit
```

**RDKit (Open Source):**
```bash
# High speed alternative  
conda install -c conda-forge rdkit
```

### ROCS Tutorial Environment

**tutorial/rocs/** contains comprehensive ROCS evaluation tools:
- Unified ROCS evaluation scripts comparing different implementations
- Performance benchmarking across CPU/GPU ROCS variants  
- Integration examples with DrugEx reinforcement learning
- Jupyter notebooks demonstrating ROCS-guided molecular generation
- **tutorial/rocs/rdkit_cdpkit/** - Complete open-source workflows with pre-trained models

## Key Architecture Patterns

### Generator Architecture
DrugEx uses a multi-generator approach supporting:
- **RNN Models**: GRU and LSTM-based sequence generators
- **Transformer Models**: Both SMILES-based and graph-based transformers
- **Fragment-based Generation**: BRICS and RECAP fragmentation strategies

### Multi-objective Reinforcement Learning
The training system implements Pareto-based multi-objective optimization:
- Multiple scoring functions can be combined
- Exploration vs exploitation balance through various explorer strategies
- Support for custom reward functions and molecular property optimization

### Scorer System
Highly modular scoring system supporting:
- **QSAR Models**: Integration with QSPRPred for QSAR-based scoring
- **Shape-based Scoring**: Multiple ROCS implementations for molecular similarity
- **Property Scoring**: Molecular descriptors, drug-likeness, synthetic accessibility
- **Custom Scorers**: Easy integration of new scoring functions

## File Structure Notes

- Configuration is handled through `pyproject.toml` with setuptools backend
- Package metadata and dependencies are defined in `pyproject.toml`
- CLI entry point is `scripts/drugex` which delegates to Python modules
- Tests are distributed throughout the codebase (not centralized)
- Tutorial notebooks are in `tutorial/` with model-specific subdirectories
- The current branch `multitask_rocs` focuses on ROCS scorer enhancements

## Dependencies

Core dependencies include PyTorch, RDKit, scikit-learn, and numpy. The optional QSPRPred dependency adds QSAR model support. ROCS functionality requires OpenEye toolkit licenses for full capability, though open-source alternatives (CDPKit and RDKit) provide license-free options.

## ROCS Troubleshooting

### Common Issues and Solutions

**OpenEye License Issues:**
```bash
# Check license status
python -c "from openeye import oechem; print('License valid:', oechem.OEChemIsLicensed())"

# Set license environment variable
export OE_LICENSE=/path/to/oe_license.txt
# or
export OE_LICENSE=server_name:port
```

**Memory Issues with Large Datasets:**
- Use `SmartCLIROCSScorer` for memory-optimized OpenEye ROCS
- Reduce batch sizes: `batch_size=200, rocs_batch_size=16`
- Enable memory optimization flags in training scripts
- Use CDPKit or RDKit alternatives for memory-constrained systems

**CDPKit Installation Issues:**
```bash
# Install from conda-forge (recommended)
conda install -c conda-forge cdpkit

# If import fails, check environment
python -c "import CDPL.Chem; print('CDPKit working')"
```

**Performance Optimization:**
- **GPU Systems**: Use OpenEye FastROCS with `use_gpu=True`
- **CPU Systems**: Use RDKit scorers for best speed
- **High Accuracy**: Use CDPKit scorers for research applications
- **Memory Limited**: Use RDKit with reduced conformer counts

## Reference Documentation

Comprehensive reference documentation is available in the `/ref` directory:

### Core System Documentation
- **[Core Modules Reference](ref/core-modules.md)** - Detailed documentation of key DrugEx modules including training pipeline, generation, dataset processing, and RL environment
- **[Architecture Reference](ref/architecture.md)** - Complete architectural overview, design patterns, multi-objective optimization, and extension points

### ROCS Integration Documentation  
- **[ROCS Scorers Reference](ref/rocs-scorers.md)** - Comprehensive guide to all ROCS scoring implementations with performance characteristics, configuration options, and decision matrix for choosing the right scorer

### Usage Documentation
- **[Tutorials Reference](ref/tutorials.md)** - Complete guide to all available tutorials from beginner RNN tutorials to advanced ROCS integration workflows
- **[CLI Usage Reference](ref/cli-usage.md)** - Comprehensive command-line interface documentation with examples, configuration options, and troubleshooting

These reference files provide detailed information for developers working with specific aspects of the DrugEx system. Refer to them for comprehensive documentation beyond the quick-start information in this file.