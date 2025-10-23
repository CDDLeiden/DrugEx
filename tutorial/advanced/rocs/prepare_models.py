#!/usr/bin/env python3
"""
Prepare Fine-Tuned Models for ROCS-Based RL

This script handles the fine-tuning step of the DrugEx workflow:
1. Load and standardize CCR2 ligand data
2. Encode SMILES using pretrained vocabulary
3. Split into train/test sets
4. Fine-tune pretrained model on CCR2 data

Usage:
    python prepare_models.py [--epochs EPOCHS] [--batch-size BATCH_SIZE]

The fine-tuned model will be saved to demo_out/models/ and can be used
for the reinforcement learning tutorial notebooks.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.processing import Standardization, CorpusEncoder, RandomTrainTestSplitter
from drugex.data.corpus.corpus import SequenceCorpus
from drugex.data.datasets import SmilesDataSet
from drugex.training.generators import SequenceRNN
from drugex.training.monitors import FileMonitor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fine-tune DrugEx model on CCR2 data')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of fine-tuning epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for training (default: 256)')
    parser.add_argument('--n-processes', type=int, default=20,
                        help='Number of CPU cores to use (default: 20)')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience (default: 30)')
    parser.add_argument('--force', action='store_true',
                        help='Force retraining even if model exists')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Define paths
    ROOT = Path.cwd()
    RES = ROOT / 'rocs_rl_ccr/rdkit_cdpkit'
    CCR2_TSV = RES / 'CCR_HUMAN_AL.tsv'
    MODEL_DIR = ROOT / 'demo_out/models'
    DATA_DIR = ROOT / 'demo_out/datasets/encoded/rnn/'
    MODELS_PR_PATH = "../../data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT/"

    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Model output paths
    FINETUNE_BASE = MODEL_DIR / 'CCR2_finetuned'
    FINETUNE_CHECKPOINT = Path(f"{FINETUNE_BASE}.pkg")
    FINETUNE_VOCAB = Path(f"{FINETUNE_BASE}.vocab")

    # Check if model already exists
    if FINETUNE_CHECKPOINT.exists() and not args.force:
        print("=" * 60)
        print("FINE-TUNED MODEL ALREADY EXISTS")
        print("=" * 60)
        print(f"Found existing model at: {FINETUNE_CHECKPOINT}")
        print("Use --force to retrain")
        return

    # =========================================================================
    # Step 1: Load and Standardize CCR2 Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading and Standardizing CCR2 Ligand Data")
    print("=" * 60)

    if not CCR2_TSV.exists():
        raise FileNotFoundError(f"CCR2 data not found at: {CCR2_TSV}")

    df_ccr2 = pd.read_table(CCR2_TSV)
    print(f"Loaded {len(df_ccr2)} CCR2 ligands")

    standardizer = Standardization(n_proc=args.n_processes, chunk_size=1000)
    smiles_ft = standardizer.apply(df_ccr2.SMILES)
    print(f"Standardized {len(smiles_ft)} SMILES")

    # =========================================================================
    # Step 2: Encode SMILES with Pretrained Vocabulary
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Encoding SMILES")
    print("=" * 60)

    voc_path = f"{MODELS_PR_PATH}/Papyrus05.5_smiles_rnn_PT.vocab"
    if not Path(voc_path).exists():
        raise FileNotFoundError(f"Pretrained vocabulary not found at: {voc_path}")

    voc = VocSmiles.fromFile(voc_path, encode_frags=False)
    print(f"Loaded vocabulary with {voc.size} tokens")

    encoder = CorpusEncoder(
        SequenceCorpus,
        {
            'vocabulary': voc,
            'update_voc': False,  # Keep vocabulary fixed
            'throw': True  # Discard compounds with unknown tokens
        },
        n_proc=args.n_processes,
        chunk_size=1000
    )

    data_collector = SmilesDataSet(
        os.path.join(DATA_DIR, 'ccr2_ligand_corpus.tsv'),
        rewrite=True
    )
    encoder.apply(smiles_ft, collector=data_collector)
    print(f"Encoded {len(data_collector.getData())} molecules")

    # =========================================================================
    # Step 3: Split into Train/Test Sets
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Creating Train/Test Split")
    print("=" * 60)

    splitter = RandomTrainTestSplitter(0.05, 1e4)
    train, test = splitter(data_collector.getData())

    for data, name in zip([train, test], ['train', 'test']):
        file_path = os.path.join(DATA_DIR, f'ccr2_{name}.tsv')
        pd.DataFrame(data).to_csv(file_path, header=True, index=False, sep='\t')
        print(f"Saved {len(data)} molecules to {name} set")

    # Create data loaders
    data_set_train = SmilesDataSet(os.path.join(DATA_DIR, 'ccr2_train.tsv'), voc=voc)
    data_set_test = SmilesDataSet(os.path.join(DATA_DIR, 'ccr2_test.tsv'), voc=voc)
    loader_train = data_set_train.asDataLoader(batch_size=args.batch_size)
    loader_test = data_set_test.asDataLoader(batch_size=args.batch_size)

    print(f"Train batches: {len(loader_train)}, Test batches: {len(loader_test)}")

    # =========================================================================
    # Step 4: Fine-Tune Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Fine-Tuning Pretrained Model on CCR2 Data")
    print("=" * 60)

    pretrained_path = os.path.join(MODELS_PR_PATH, 'Papyrus05.5_smiles_rnn_PT.pkg')
    if not Path(pretrained_path).exists():
        raise FileNotFoundError(f"Pretrained model not found at: {pretrained_path}")

    finetuned = SequenceRNN(voc, is_lstm=True)
    finetuned.loadStatesFromFile(pretrained_path)
    print(f"Loaded pretrained model from {pretrained_path}")

    monitor = FileMonitor(str(FINETUNE_BASE), save_smiles=True, reset_directory=True)

    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    print(f"Using device: {finetuned.device}")

    start = time.time()
    finetuned.fit(
        loader_train,
        loader_test,
        epochs=args.epochs,
        monitor=monitor,
        patience=args.patience
    )
    monitor.close()

    elapsed = time.time() - start

    # Save model
    torch.save(finetuned.getModel(), str(FINETUNE_CHECKPOINT))
    voc.toFile(str(FINETUNE_VOCAB))

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Model saved to: {FINETUNE_CHECKPOINT}")
    print(f"Vocabulary saved to: {FINETUNE_VOCAB}")

    # Show final metrics
    fit_file = f'{FINETUNE_BASE}_fit.tsv'
    if os.path.exists(fit_file):
        df_metrics = pd.read_csv(fit_file, sep='\t')
        print("\nFinal Training Metrics:")
        print(df_metrics[['loss_train', 'loss_valid']].tail(5))

    print("\nYou can now run the RL tutorial notebooks!")


if __name__ == '__main__':
    main()
