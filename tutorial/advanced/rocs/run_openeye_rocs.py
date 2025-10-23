#!/usr/bin/env python3
"""
DrugEx RL with OpenEye ROCS CLI Shape-Based Scoring

This script demonstrates using the commercial OpenEye ROCS software
for shape-based scoring in DrugEx reinforcement learning.

Key features of OpenEye ROCS:
- Industry-standard commercial implementation
- Highly optimized algorithms
- GPU acceleration support
- Advanced color force field for pharmacophore matching
- Shape optimization during overlay

Prerequisites:
    - Run prepare_models.py first to create fine-tuned model
    - OpenEye ROCS license and 'rocs' binary in PATH
    - OpenEye Omega for conformer generation

Usage:
    python run_openeye_rocs.py [--epochs EPOCHS] [--samples SAMPLES]
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.generators import SequenceRNN
from drugex.training.monitors import FileMonitor
from drugex.training.explorers import SequenceExplorer
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.scorers.properties import Property

try:
    from drugex.training.scorers.conformer_generators import OmegaConformerGenerator
    from drugex.training.scorers.cli_rocs import CLIROCSScorer
    OPENEYE_READY = True
except ImportError:
    print("ERROR: OpenEye not available!")
    print("Install OpenEye toolkit and configure license")
    OPENEYE_READY = False

ROCS_BINARY = shutil.which('rocs') if OPENEYE_READY else None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RL with OpenEye ROCS scoring')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of RL epochs (default: 30)')
    parser.add_argument('--samples', type=int, default=500,
                        help='Molecules per epoch (default: 500)')
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help='Exploration rate (default: 0.2)')
    parser.add_argument('--max-conformers', type=int, default=30,
                        help='Max conformers per molecule (default: 30)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration for conformer generation')
    parser.add_argument('--optimize', action='store_true', default=True,
                        help='Optimize shape overlay (default: True)')
    return parser.parse_args()


def create_environment(ccr2_sdf, max_conformers=30, use_gpu=False, optimize=True):
    """
    Create DrugEx environment with OpenEye CLI ROCS scorer

    OpenEye-specific features:
    - shape_only=False: Include color (pharmacophore) scoring
    - optimize=True: Optimize overlay during shape matching
    - color_optimize=True: Optimize color force field contributions
    - GPU support for conformer generation (requires GPU license)
    """
    if not (OPENEYE_READY and ROCS_BINARY):
        raise RuntimeError("OpenEye ROCS CLI not available")

    # OpenEye ROCS scorer
    rocs_scorer = CLIROCSScorer(
        conformer_generator=OmegaConformerGenerator(
            max_conformers=max_conformers,
            max_centers=2,              # Stereoisomer enumeration
            max_heavy_atoms=40,
            use_gpu=use_gpu,            # GPU acceleration if available
            show_progress=True,
        ),
        query_files={'CCR2': str(ccr2_sdf)},
        score_type='TanimotoCombo',     # Combined shape + color
        shape_only=False,               # Include pharmacophore features
        optimize=optimize,              # Optimize shape overlay
        color_optimize=True,            # Optimize color contributions
        rocs_binary=ROCS_BINARY,
        show_progress=True,
    )

    # Synthetic accessibility scorer
    sa_scorer = Property('SA')
    sa_scorer.setModifier(SmoothClippedScore(lower_x=5, upper_x=3))

    # Create environment
    env = DrugExEnvironment(
        scorers=[rocs_scorer, sa_scorer],
        weights=[0.9, 0.1],
        reward_scheme=ParetoCrowdingDistance()
    )

    return env


def main():
    if not (OPENEYE_READY and ROCS_BINARY):
        print("OpenEye ROCS not available. Please install and configure:")
        print("  1. Install OpenEye toolkit")
        print("  2. Set up valid license")
        print("  3. Ensure 'rocs' is in your PATH")
        return

    args = parse_args()

    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Define paths
    ROOT = Path.cwd()
    CCR2_SDF = ROOT / 'rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf'
    MODEL_DIR = ROOT / 'demo_out/models'
    OUTPUT_DIR = ROOT / 'rl_runs_demo/openeye_rl'
    MODELS_PR_PATH = "../../data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT/"

    FINETUNE_BASE = MODEL_DIR / 'CCR2_finetuned'
    FINETUNE_CHECKPOINT = Path(f"{FINETUNE_BASE}.pkg")
    FINETUNE_VOCAB = Path(f"{FINETUNE_BASE}.vocab")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify prerequisites
    if not FINETUNE_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Fine-tuned model not found! Run prepare_models.py first"
        )

    print("=" * 60)
    print("DrugEx RL with OpenEye ROCS CLI")
    print("=" * 60)
    print(f"ROCS binary: {ROCS_BINARY}")
    print(f"RL epochs: {args.epochs}")
    print(f"Samples per epoch: {args.samples}")
    print(f"Exploration rate (epsilon): {args.epsilon}")
    print(f"Max conformers: {args.max_conformers}")
    print(f"GPU acceleration: {args.use_gpu}")
    print(f"Shape optimization: {args.optimize}")
    print()

    # Load vocabulary
    voc = VocSmiles.fromFile(str(FINETUNE_VOCAB), encode_frags=False)
    print(f"Loaded vocabulary: {voc.size} tokens")

    # Create environment
    print("\nSetting up OpenEye ROCS environment...")
    env = create_environment(
        CCR2_SDF,
        max_conformers=args.max_conformers,
        use_gpu=args.use_gpu,
        optimize=args.optimize
    )

    # Initialize models
    print("Loading models...")
    agent = SequenceRNN(voc, is_lstm=True)
    agent.loadStatesFromFile(os.path.join(MODELS_PR_PATH, 'Papyrus05.5_smiles_rnn_PT.pkg'))

    mutate = SequenceRNN(voc, is_lstm=True)
    mutate.loadStatesFromFile(str(FINETUNE_CHECKPOINT))

    # Create explorer
    explorer = SequenceExplorer(
        agent=agent,
        env=env,
        mutate=mutate,
        epsilon=args.epsilon,
        n_samples=args.samples,
    )

    # Train
    print(f"\nStarting RL training with OpenEye ROCS...")
    print(f"Using device: {agent.device}\n")

    output_base = OUTPUT_DIR / 'CCR2_openeye_reinforced'
    monitor = FileMonitor(str(output_base), save_smiles=True, reset_directory=True)

    start_time = time.time()
    explorer.fit(monitor=monitor, epochs=args.epochs)
    monitor.close()

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("RL TRAINING COMPLETE")
    print("=" * 60)
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Model saved to: {output_base}.pkg")

    # Show metrics
    fit_file = f'{output_base}_fit.tsv'
    if os.path.exists(fit_file):
        df_metrics = pd.read_csv(fit_file, sep='\\t')
        print("\nFinal Metrics (last 5 epochs):")
        print(df_metrics[['epoch', 'desired_ratio', 'avg_amean']].tail())

    # Generate molecules
    print("\n" + "=" * 60)
    print("GENERATING MOLECULES")
    print("=" * 60)

    agent_rl = SequenceRNN(voc, is_lstm=True)
    agent_rl.loadStatesFromFile(str(output_base) + '.pkg')

    df_gen = agent_rl.generate(num_samples=100, evaluator=env)
    scorer_keys = env.getScorerKeys()
    df_gen['Total'] = df_gen[scorer_keys].mean(axis=1)

    output_file = f'{output_base}_generated.tsv'
    df_gen.to_csv(output_file, sep='\\t', index=False)

    print(f"Generated {len(df_gen)} molecules")
    print(f"Saved to: {output_file}")

    print("\nScore Statistics:")
    print(df_gen[scorer_keys + ['Total']].describe())

    if 'Desired' in df_gen.columns:
        n_desired = int(df_gen['Desired'].sum())
        pct_desired = df_gen['Desired'].mean() * 100
        print(f"\nDesired molecules: {n_desired} / {len(df_gen)} ({pct_desired:.1f}%)")

    print("\nTop 5 molecules by Total Score:")
    print(df_gen.nlargest(5, 'Total')[['SMILES'] + scorer_keys + ['Total']])

    print("\n" + "=" * 60)
    print("OPENEYE ROCS NOTES")
    print("=" * 60)
    print("OpenEye advantages:")
    print("  - Industry-standard commercial implementation")
    print("  - Highly optimized shape matching algorithms")
    print("  - GPU acceleration support")
    print("  - Advanced color force field")
    print("\nPerformance tips:")
    print("  - Use --use-gpu for faster conformer generation")
    print("  - Reduce --max-conformers for speed")
    print("  - Use --no-optimize to skip shape optimization")
    print("\nFor comparison with open-source backends:")
    print("  - RDKit: python quickstart_rl_rdkit.ipynb")
    print("  - CDPKit: python run_cdpkit_rocs.py")


if __name__ == '__main__':
    main()
