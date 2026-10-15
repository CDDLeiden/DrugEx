#!/usr/bin/env python3
"""DrugEx RL with OpenEye ROCS shape-based scoring.

OpenEye ROCS is the industry-standard commercial implementation offering
highly optimized shape matching algorithms, GPU acceleration, and advanced
color force field for pharmacophore matching.

Prerequisites:
    - Fine-tuned model (run prepare_models.py first)
    - OpenEye ROCS license and 'rocs' binary in PATH
    - OpenEye Omega for conformer generation

References:
    Hawkins et al. "Comparison of Shape-Matching and Docking as Virtual
    Screening Tools." J Med Chem 2007, 50(1), 74-82.
"""

import argparse
import shutil
import time
from pathlib import Path
from typing import Tuple

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
    from drugex.training.scorers.rocs_openeye import OpenEyeROCSScorer
    OPENEYE_AVAILABLE = True
except ImportError:
    print("OpenEye not available. Install OpenEye toolkit and configure license")
    OPENEYE_AVAILABLE = False

ROCS_BINARY = shutil.which('rocs') if OPENEYE_AVAILABLE else None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='RL training with OpenEye ROCS scoring'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of RL epochs (default: 30)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=500,
        help='Molecules generated per epoch (default: 500)'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.2,
        help='Exploration rate (default: 0.2)'
    )
    parser.add_argument(
        '--max-conformers',
        type=int,
        default=30,
        help='Max conformers per molecule (default: 30)'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU acceleration for conformer generation'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        default=True,
        help='Optimize shape overlay (default: True)'
    )
    return parser.parse_args()


def create_openeye_environment(
    reference_sdf: Path,
    max_conformers: int = 30,
    use_gpu: bool = False,
    optimize: bool = True
) -> DrugExEnvironment:
    """Create DrugEx environment with OpenEye CLI ROCS scorer.

    Args:
        reference_sdf: Path to reference ligands SDF file.
        max_conformers: Maximum conformers per molecule.
        use_gpu: Enable GPU acceleration.
        optimize: Optimize overlay during shape matching.

    Returns:
        Configured DrugExEnvironment.

    Raises:
        RuntimeError: If OpenEye ROCS is not available.
    """
    if not (OPENEYE_AVAILABLE and ROCS_BINARY):
        raise RuntimeError("OpenEye ROCS CLI not available")

    rocs_scorer = OpenEyeROCSScorer(
        conformer_generator=OmegaConformerGenerator(
            max_conformers=max_conformers,
            max_centers=2,
            max_heavy_atoms=40,
            use_gpu=use_gpu,
            show_progress=False,
        ),
        references={'CCR2': str(reference_sdf)},
        score_type='TanimotoCombo',
        shape_only=False,
        optimize=optimize,
        color_optimize=True,
        rocs_binary=ROCS_BINARY,
        show_progress=False,
    )

    sa_scorer = Property('SA')
    sa_scorer.setModifier(SmoothClippedScore(lower_x=5, upper_x=3))

    env = DrugExEnvironment(
        scorers=[rocs_scorer, sa_scorer],
        thresholds=[0.9, 0.1],
        reward_scheme=ParetoCrowdingDistance()
    )

    return env


def run_rl_training(
    agent: SequenceRNN,
    mutate: SequenceRNN,
    env: DrugExEnvironment,
    output_path: Path,
    epochs: int,
    epsilon: float,
    n_samples: int
) -> Tuple[float, pd.DataFrame]:
    """Run reinforcement learning training.

    Args:
        agent: Pretrained agent network.
        mutate: Fine-tuned mutate network.
        env: Environment for scoring.
        output_path: Path for saving results.
        epochs: Number of training epochs.
        epsilon: Exploration rate.
        n_samples: Molecules per epoch.

    Returns:
        Tuple of (training_time, metrics_dataframe).
    """
    explorer = SequenceExplorer(
        agent=agent,
        env=env,
        mutate=mutate,
        epsilon=epsilon,
        n_samples=n_samples,
    )

    monitor = FileMonitor(str(output_path), save_smiles=True, reset_directory=True)

    start_time = time.time()
    explorer.fit(monitor=monitor, epochs=epochs)
    monitor.close()
    elapsed = time.time() - start_time

    # Load metrics if available
    metrics_file = output_path.with_name(f'{output_path.name}_fit.tsv')
    metrics_df = None
    if metrics_file.exists():
        metrics_df = pd.read_csv(metrics_file, sep='\t')

    return elapsed, metrics_df


def generate_molecules(
    model_path: Path,
    voc: VocSmiles,
    env: DrugExEnvironment,
    num_samples: int = 100
) -> pd.DataFrame:
    """Generate molecules from trained model.

    Args:
        model_path: Path to trained model checkpoint.
        voc: Vocabulary object.
        env: Environment for scoring.
        num_samples: Number of molecules to generate.

    Returns:
        DataFrame with generated molecules and scores.
    """
    agent = SequenceRNN(voc, is_lstm=True)
    agent.loadStatesFromFile(str(model_path))

    df_gen = agent.generate(num_samples=num_samples, evaluator=env)
    scorer_keys = env.getScorerKeys()
    df_gen['Total'] = df_gen[scorer_keys].mean(axis=1)

    return df_gen


def main() -> None:
    """Main execution function."""
    if not (OPENEYE_AVAILABLE and ROCS_BINARY):
        print("OpenEye ROCS not available.")
        print("Installation steps:")
        print("  1. Install OpenEye toolkit")
        print("  2. Configure valid license")
        print("  3. Ensure 'rocs' is in PATH")
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
    PRETRAINED_DIR = Path("../../data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT")

    FINETUNE_BASE = MODEL_DIR / 'CCR2_finetuned'
    FINETUNE_CHECKPOINT = FINETUNE_BASE.with_suffix('.pkg')
    FINETUNE_VOCAB = FINETUNE_BASE.with_suffix('.vocab')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify prerequisites
    if not FINETUNE_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Fine-tuned model not found. Run prepare_models.py first"
        )

    print("DrugEx RL with OpenEye ROCS")
    print(f"Epochs: {args.epochs}, Samples/epoch: {args.samples}, Epsilon: {args.epsilon}")
    print(f"Conformers: {args.max_conformers}, GPU: {args.use_gpu}, Optimize: {args.optimize}")

    # Load vocabulary
    voc = VocSmiles.fromFile(str(FINETUNE_VOCAB), encode_frags=False)
    print(f"\nVocabulary: {voc.size} tokens")

    # Create environment
    print("Setting up OpenEye ROCS environment...")
    env = create_openeye_environment(
        CCR2_SDF,
        max_conformers=args.max_conformers,
        use_gpu=args.use_gpu,
        optimize=args.optimize
    )

    # Initialize models
    print("Loading models...")
    agent = SequenceRNN(voc, is_lstm=True)
    agent.loadStatesFromFile(str(PRETRAINED_DIR / 'Papyrus05.5_smiles_rnn_PT.pkg'))

    mutate = SequenceRNN(voc, is_lstm=True)
    mutate.loadStatesFromFile(str(FINETUNE_CHECKPOINT))
    print(f"Device: {agent.device}")

    # Train
    print(f"\nStarting RL training ({args.epochs} epochs)...")
    output_base = OUTPUT_DIR / 'CCR2_openeye_reinforced'

    training_time, metrics_df = run_rl_training(
        agent,
        mutate,
        env,
        output_base,
        args.epochs,
        args.epsilon,
        args.samples
    )

    print(f"\nRL training complete in {training_time/60:.1f} minutes")
    print(f"Model saved: {output_base}.pkg")

    if metrics_df is not None:
        final = metrics_df[['epoch', 'desired_ratio', 'avg_amean']].tail(3)
        print(f"\nFinal epochs:\n{final.to_string(index=False)}")

    # Generate molecules
    print("\nGenerating molecules...")
    df_gen = generate_molecules(
        output_base.with_suffix('.pkg'),
        voc,
        env,
        num_samples=100
    )

    output_file = output_base.with_name(f'{output_base.name}_generated.tsv')
    df_gen.to_csv(output_file, sep='\t', index=False)

    scorer_keys = env.getScorerKeys()
    stats = df_gen[scorer_keys + ['Total']].describe().loc[['mean', 'max']]
    print(f"Generated {len(df_gen)} molecules")
    print(f"\nScore summary:\n{stats.to_string()}")

    if 'Desired' in df_gen.columns:
        n_desired = int(df_gen['Desired'].sum())
        print(f"Desired: {n_desired}/{len(df_gen)} ({100*n_desired/len(df_gen):.1f}%)")

    print(f"\nResults saved: {output_file}")


if __name__ == '__main__':
    main()
