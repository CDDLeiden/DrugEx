#!/usr/bin/env python3
"""Generate molecules from trained RL model.

This script loads a reinforcement learning-trained model and generates
novel molecules optimized for specified objectives (ROCS shape similarity
and synthetic accessibility).
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.generators import SequenceRNN
from config import create_rdkit_environment, FINETUNE_VOCAB, OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
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


def load_model(model_path: Path, voc: VocSmiles) -> SequenceRNN:
    """Load trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint file.
        voc: Vocabulary object for model.

    Returns:
        Loaded SequenceRNN model.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train a model first with: jupyter notebook rocs_rl_tutorial.ipynb"
        )

    agent = SequenceRNN(voc, is_lstm=True)
    agent.loadStatesFromFile(str(model_path))
    return agent


def generate_and_score(
    agent: SequenceRNN,
    env,
    num_samples: int
) -> pd.DataFrame:
    """Generate molecules and score them with environment.

    Args:
        agent: Trained SequenceRNN model.
        env: DrugExEnvironment for scoring.
        num_samples: Number of molecules to generate.

    Returns:
        DataFrame with generated molecules and scores.
    """
    print(f"Generating {num_samples} molecules...")
    df_gen = agent.generate(num_samples=num_samples, evaluator=env)

    # Calculate total score
    scorer_keys = env.getScorerKeys()
    df_gen['Total'] = df_gen[scorer_keys].mean(axis=1)

    return df_gen


def print_summary(df: pd.DataFrame, scorer_keys: list) -> None:
    """Print summary statistics and top molecules.

    Args:
        df: DataFrame with generated molecules and scores.
        scorer_keys: List of scorer column names.
    """
    print("\nScore Statistics:")
    stats = df[scorer_keys + ['Total']].describe()
    print(stats.loc[['mean', '50%', 'max']])

    # Count desired molecules
    if 'Desired' in df.columns:
        n_desired = int(df['Desired'].sum())
        pct_desired = df['Desired'].mean() * 100
        print(f"\nDesired molecules: {n_desired}/{len(df)} ({pct_desired:.1f}%)")

    # Show top molecules
    print("\nTop 5 Molecules by Total Score:")
    top5 = df.nlargest(5, 'Total')[['SMILES'] + scorer_keys + ['Total']]
    for idx, row in top5.iterrows():
        print(f"\n{row['SMILES']}")
        for key in scorer_keys + ['Total']:
            print(f"  {key}: {row[key]:.3f}")


def main() -> None:
    """Main execution function."""
    args = parse_args()

    model_path = Path(args.model)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = model_path.parent / f"{model_path.stem}_generated.tsv"

    print("Molecule Generation")
    print(f"Model: {model_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {output_path}")

    # Load vocabulary
    voc = VocSmiles.fromFile(str(FINETUNE_VOCAB), encode_frags=False)
    print(f"\nVocabulary loaded: {voc.size} tokens")

    # Create environment for scoring
    env = create_rdkit_environment()
    print("Environment created for scoring")

    # Load trained model
    agent = load_model(model_path, voc)
    print(f"Model loaded (device: {agent.device})")

    # Generate and score molecules
    df_gen = generate_and_score(agent, env, args.num_samples)

    # Save results
    df_gen.to_csv(output_path, sep='\t', index=False)
    print(f"\nSaved {len(df_gen)} molecules to: {output_path}")

    # Print statistics
    scorer_keys = env.getScorerKeys()
    print_summary(df_gen, scorer_keys)

    print("\nGeneration complete")


if __name__ == '__main__':
    main()
