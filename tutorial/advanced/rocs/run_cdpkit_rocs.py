#!/usr/bin/env python3
"""
DrugEx RL with CDPKit ROCS Shape-Based Scoring

This script demonstrates using the CDPKit library for ROCS shape-based scoring
in DrugEx reinforcement learning. CDPKit is an open-source alternative to
OpenEye's commercial ROCS implementation.

Key features of CDPKit ROCS:
- Open-source implementation (no license required)
- Multi-conformer shape generation
- Optimized for performance with parallel processing
- Good agreement with commercial ROCS implementations

Prerequisites:
    - Run prepare_models.py first to create fine-tuned model
    - CDPKit must be installed (pip install drugex[cdpkit])

Usage:
    python run_cdpkit_rocs.py [--epochs EPOCHS] [--samples SAMPLES]
"""

import argparse
import os
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
    from drugex.training.scorers.conformer_generators import CDPKitConformerGenerator
    from drugex.training.scorers.cdpkit_rocs import CDPKitROCSScorer
    CDPKIT_READY = True
except ImportError:
    print("ERROR: CDPKit not available!")
    print("Install with: pip install drugex[cdpkit]")
    CDPKIT_READY = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RL with CDPKit ROCS scoring')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of RL epochs (default: 30)')
    parser.add_argument('--samples', type=int, default=500,
                        help='Molecules per epoch (default: 500)')
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help='Exploration rate (default: 0.2)')
    parser.add_argument('--max-conformers', type=int, default=30,
                        help='Max conformers per molecule (default: 30)')
    parser.add_argument('--max-isomers', type=int, default=4,
                        help='Max stereoisomers to consider (default: 4)')
    return parser.parse_args()


def create_environment(ccr2_sdf, max_conformers=30, max_isomers=4):
    """
    Create DrugEx environment with CDPKit ROCS scorer

    CDPKit-specific features:
    - max_isomers: CDPKit can enumerate and score multiple stereoisomers
    - Optimized conformer generation with native CDPKit algorithms
    - Parallel processing for shape overlays
    """
    if not CDPKIT_READY:
        raise RuntimeError("CDPKit bindings not available")

    # CDPKit ROCS scorer
    rocs_scorer = CDPKitROCSScorer(
        conformer_generator=CDPKitConformerGenerator(
            max_conformers=max_conformers,
            max_isomers=max_isomers,     # CDPKit can handle multiple isomers
            max_heavy_atoms=40,
            show_progress=True,
        ),
        references=str(ccr2_sdf),
        show_progress=True,
        n_jobs=-1,
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
    if not CDPKIT_READY:
        return

    args = parse_args()

    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Define paths
    ROOT = Path.cwd()
    CCR2_SDF = ROOT / 'rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf'
    MODEL_DIR = ROOT / 'demo_out/models'
    OUTPUT_DIR = ROOT / 'rl_runs_demo/cdpkit_rl'
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
    print("DrugEx RL with CDPKit ROCS")
    print("=" * 60)
    print(f"RL epochs: {args.epochs}")
    print(f"Samples per epoch: {args.samples}")
    print(f"Exploration rate (epsilon): {args.epsilon}")
    print(f"Max conformers: {args.max_conformers}")
    print(f"Max isomers: {args.max_isomers}")
    print()

    # Load vocabulary
    voc = VocSmiles.fromFile(str(FINETUNE_VOCAB), encode_frags=False)
    print(f"Loaded vocabulary: {voc.size} tokens")

    # Create environment
    print("\nSetting up CDPKit ROCS environment...")
    env = create_environment(
        CCR2_SDF,
        max_conformers=args.max_conformers,
        max_isomers=args.max_isomers
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
    print(f"\nStarting RL training with CDPKit ROCS...")
    print(f"Using device: {agent.device}\n")

    output_base = OUTPUT_DIR / 'CCR2_cdpkit_reinforced'
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
    print("CDPKit ROCS NOTES")
    print("=" * 60)
    print("CDPKit advantages:")
    print("  - Open-source (no license required)")
    print("  - Multi-isomer scoring capability")
    print("  - Good performance with parallel processing")
    print("\nFor comparison with other backends:")
    print("  - RDKit: python quickstart_rl_rdkit.ipynb")
    print("  - OpenEye: python run_openeye_rocs.py")


if __name__ == '__main__':
    main()
