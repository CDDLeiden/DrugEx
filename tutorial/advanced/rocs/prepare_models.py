#!/usr/bin/env python3
"""Fine-tune pretrained DrugEx model on CCR2 ligand data.

This script prepares models for ROCS-based reinforcement learning by:
1. Loading and standardizing CCR2 ligand SMILES
2. Encoding with pretrained vocabulary
3. Creating train/test split
4. Fine-tuning pretrained RNN model

The fine-tuned model enables domain-specific molecule generation in RL tutorials.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.processing import (
    Standardization,
    CorpusEncoder,
    RandomTrainTestSplitter
)
from drugex.data.corpus.corpus import SequenceCorpus
from drugex.data.datasets import SmilesDataSet
from drugex.training.generators import SequenceRNN
from drugex.training.monitors import FileMonitor


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Fine-tune DrugEx model on CCR2 ligand data'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of fine-tuning epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for training (default: 256)'
    )
    parser.add_argument(
        '--n-processes',
        type=int,
        default=20,
        help='Number of CPU cores to use (default: 20)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=30,
        help='Early stopping patience (default: 30)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force retraining even if model exists'
    )
    return parser.parse_args()


def load_and_standardize_data(
    data_path: Path,
    n_proc: int,
    chunk_size: int = 1000
) -> list:
    """Load and standardize SMILES from TSV file.

    Args:
        data_path: Path to TSV file with 'SMILES' column.
        n_proc: Number of parallel processes.
        chunk_size: Chunk size for parallel processing.

    Returns:
        List of standardized SMILES strings.

    Raises:
        FileNotFoundError: If data file doesn't exist.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"CCR2 data not found: {data_path}")

    df = pd.read_table(data_path)
    print(f"Loaded {len(df)} CCR2 ligands from {data_path.name}")

    standardizer = Standardization(n_proc=n_proc, chunk_size=chunk_size)
    standardized_smiles = standardizer.apply(df.SMILES)
    print(f"Standardized {len(standardized_smiles)} SMILES")

    return standardized_smiles


def encode_smiles(
    smiles_list: list,
    voc: VocSmiles,
    output_path: Path,
    n_proc: int,
    chunk_size: int = 1000
) -> SmilesDataSet:
    """Encode SMILES using vocabulary.

    Args:
        smiles_list: List of SMILES strings to encode.
        voc: Vocabulary object for encoding.
        output_path: Path for saving encoded corpus.
        n_proc: Number of parallel processes.
        chunk_size: Chunk size for parallel processing.

    Returns:
        SmilesDataSet containing encoded molecules.
    """
    encoder = CorpusEncoder(
        SequenceCorpus,
        {
            'vocabulary': voc,
            'update_voc': False,  # Keep vocabulary fixed
            'throw': True  # Discard compounds with unknown tokens
        },
        n_proc=n_proc,
        chunk_size=chunk_size
    )

    data_collector = SmilesDataSet(str(output_path), rewrite=True)
    encoder.apply(smiles_list, collector=data_collector)
    print(f"Encoded {len(data_collector.getData())} molecules")

    return data_collector


def create_train_test_split(
    dataset: SmilesDataSet,
    output_dir: Path,
    test_fraction: float = 0.05,
    max_test_size: int = 10000
) -> Tuple[DataLoader, DataLoader]:
    """Create train/test split and data loaders.

    Args:
        dataset: Input dataset to split.
        output_dir: Directory for saving split datasets.
        test_fraction: Fraction of data for test set.
        max_test_size: Maximum test set size.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    splitter = RandomTrainTestSplitter(test_fraction, max_test_size)
    train, test = splitter(dataset.getData())

    # Save splits
    for data, name in zip([train, test], ['train', 'test']):
        file_path = output_dir / f'ccr2_{name}.tsv'
        pd.DataFrame(data).to_csv(file_path, header=True, index=False, sep='\t')
        print(f"Saved {len(data)} molecules to {name} set")

    return train, test


def fine_tune_model(
    pretrained_path: Path,
    voc: VocSmiles,
    train_loader: DataLoader,
    test_loader: DataLoader,
    output_path: Path,
    epochs: int,
    patience: int
) -> Tuple[SequenceRNN, float]:
    """Fine-tune pretrained model on domain data.

    Args:
        pretrained_path: Path to pretrained model checkpoint.
        voc: Vocabulary object.
        train_loader: Training data loader.
        test_loader: Test data loader.
        output_path: Path for saving fine-tuned model.
        epochs: Number of training epochs.
        patience: Early stopping patience.

    Returns:
        Tuple of (fine_tuned_model, training_time).

    Raises:
        FileNotFoundError: If pretrained model doesn't exist.
    """
    if not pretrained_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")

    model = SequenceRNN(voc, is_lstm=True)
    model.loadStatesFromFile(str(pretrained_path))
    print(f"Loaded pretrained model from {pretrained_path.name}")
    print(f"Training on device: {model.device}")

    monitor = FileMonitor(str(output_path), save_smiles=True, reset_directory=True)

    start_time = time.time()
    model.fit(
        train_loader,
        test_loader,
        epochs=epochs,
        monitor=monitor,
        patience=patience
    )
    monitor.close()
    elapsed = time.time() - start_time

    return model, elapsed


def main() -> None:
    """Main execution function."""
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Define paths
    ROOT = Path(__file__).resolve().parent
    CCR2_TSV = ROOT / 'rocs_rl_ccr/rdkit_cdpkit/CCR_HUMAN_AL.tsv'
    MODEL_DIR = ROOT / 'demo_out/models'
    DATA_DIR = ROOT / 'demo_out/datasets/encoded/rnn'
    PRETRAINED_DIR = ROOT.parents[2] / "data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT"

    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Model output paths
    FINETUNE_BASE = MODEL_DIR / 'CCR2_finetuned'
    FINETUNE_CHECKPOINT = FINETUNE_BASE.with_suffix('.pkg')
    FINETUNE_VOCAB = FINETUNE_BASE.with_suffix('.vocab')

    # Check if model already exists
    if FINETUNE_CHECKPOINT.exists() and not args.force:
        print(f"Fine-tuned model already exists: {FINETUNE_CHECKPOINT}")
        print("Use --force to retrain")
        return

    print("Starting fine-tuning pipeline")
    print(f"Configuration: {args.epochs} epochs, batch_size={args.batch_size}")

    # Step 1: Load and standardize data
    print("\n[1/4] Loading and standardizing CCR2 data")
    smiles_list = load_and_standardize_data(
        CCR2_TSV,
        args.n_processes
    )

    # Step 2: Load vocabulary and encode SMILES
    print("\n[2/4] Encoding SMILES with pretrained vocabulary")
    voc_path = PRETRAINED_DIR / 'Papyrus05.5_smiles_rnn_PT.vocab'
    if not voc_path.exists():
        raise FileNotFoundError(f"Pretrained vocabulary not found: {voc_path}")

    voc = VocSmiles.fromFile(str(voc_path), encode_frags=False)
    print(f"Loaded vocabulary: {voc.size} tokens")

    corpus_path = DATA_DIR / 'ccr2_ligand_corpus.tsv'
    dataset = encode_smiles(
        smiles_list,
        voc,
        corpus_path,
        args.n_processes
    )

    # Step 3: Create train/test split
    print("\n[3/4] Creating train/test split")
    train_data, test_data = create_train_test_split(dataset, DATA_DIR)

    # Create data loaders
    train_set = SmilesDataSet(str(DATA_DIR / 'ccr2_train.tsv'), voc=voc)
    test_set = SmilesDataSet(str(DATA_DIR / 'ccr2_test.tsv'), voc=voc)
    train_loader = train_set.asDataLoader(batch_size=args.batch_size)
    test_loader = test_set.asDataLoader(batch_size=args.batch_size)

    # Step 4: Fine-tune model
    print(f"\n[4/4] Fine-tuning model for {args.epochs} epochs")
    pretrained_path = PRETRAINED_DIR / 'Papyrus05.5_smiles_rnn_PT.pkg'

    model, training_time = fine_tune_model(
        pretrained_path,
        voc,
        train_loader,
        test_loader,
        FINETUNE_BASE,
        args.epochs,
        args.patience
    )

    # Save final model and vocabulary
    torch.save(model.getModel(), str(FINETUNE_CHECKPOINT))
    voc.toFile(str(FINETUNE_VOCAB))

    # Print summary
    print(f"\nFine-tuning complete in {training_time/60:.1f} minutes")
    print(f"Model saved: {FINETUNE_CHECKPOINT}")
    print(f"Vocabulary saved: {FINETUNE_VOCAB}")

    # Show final metrics
    metrics_file = FINETUNE_BASE.with_suffix('.pkg').parent / f'{FINETUNE_BASE.name}_fit.tsv'
    if metrics_file.exists():
        df = pd.read_csv(metrics_file, sep='\t')
        final_metrics = df[['loss_train', 'loss_valid']].tail(1)
        print(f"\nFinal metrics:\n{final_metrics.to_string(index=False)}")


if __name__ == '__main__':
    main()
