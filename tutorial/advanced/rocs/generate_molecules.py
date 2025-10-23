#!/usr/bin/env python3
"""
Generate Molecules from Trained RL Model

This script loads a trained RL model and generates novel molecules
optimized for the defined objectives (ROCS shape similarity + SA).

Usage:
    python generate_molecules.py [--model MODEL_PATH] [--num-samples N] [--output OUTPUT]
"""

import argparse
from pathlib import Path

import pandas as pd

from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.generators import SequenceRNN
from config import (
    create_rdkit_environment,
    FINETUNE_VOCAB,
    OUTPUT_DIR,
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate molecules from trained RL model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=str(OUTPUT_DIR / 'CCR2_rdkit_reinforced.pkg'),
        help='Path to trained model checkpoint (default: from config)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of molecules to generate (default: 100)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: same as model with _generated.tsv)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = model_path.parent / f"{model_path.stem}_generated.tsv"

    print("=" * 60)
    print("MOLECULE GENERATION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Output: {output_path}")
    print()

    # Verify model exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train a model first with: jupyter notebook quickstart_rl_rdkit.ipynb"
        )

    # Load vocabulary
    voc = VocSmiles.fromFile(str(FINETUNE_VOCAB), encode_frags=False)
    print(f"Vocabulary loaded: {voc.size} tokens")

    # Create environment for evaluation
    env = create_rdkit_environment()
    print("Environment created for scoring")

    # Load trained model
    agent = SequenceRNN(voc, is_lstm=True)
    agent.loadStatesFromFile(str(model_path))
    print(f"Model loaded from: {model_path}")
    print(f"Using device: {agent.device}\n")

    # Generate molecules
    print(f"Generating {args.num_samples} molecules...")
    df_gen = agent.generate(num_samples=args.num_samples, evaluator=env)

    # Calculate total score
    scorer_keys = env.getScorerKeys()
    df_gen['Total'] = df_gen[scorer_keys].mean(axis=1)

    # Save results
    df_gen.to_csv(output_path, sep='\t', index=False)
    print(f"✓ Saved {len(df_gen)} molecules to: {output_path}")

    # Show statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nScore Statistics:")
    stats = df_gen[scorer_keys + ['Total']].describe()
    print(stats.loc[['mean', '50%', 'max']])

    # Count desired molecules
    if 'Desired' in df_gen.columns:
        n_desired = int(df_gen['Desired'].sum())
        pct_desired = df_gen['Desired'].mean() * 100
        print(f"\nDesired molecules: {n_desired} / {len(df_gen)} ({pct_desired:.1f}%)")

    # Show top molecules
    print("\nTop 5 Molecules by Total Score:")
    top5 = df_gen.nlargest(5, 'Total')[['SMILES'] + scorer_keys + ['Total']]
    for idx, row in top5.iterrows():
        print(f"\n{row['SMILES']}")
        for key in scorer_keys + ['Total']:
            print(f"  {key}: {row[key]:.3f}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Full results saved to: {output_path}")


if __name__ == '__main__':
    main()
