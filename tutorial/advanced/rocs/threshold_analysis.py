"""ROCS threshold determination analysis.

Scientific workflow for determining optimal TanimotoCombo threshold
for separating CCR2 active ligands from decoys using ROC analysis.
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer

warnings.filterwarnings('ignore')

try:
    from config import (
        CCR2_SDF, MAX_CONFORMERS, MAX_ISOMERS,
        MAX_HEAVY_ATOMS, MAX_ROTATABLE_BONDS, ROCS_THRESHOLD
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    MAX_CONFORMERS = 50
    MAX_ISOMERS = 4
    MAX_HEAVY_ATOMS = 45
    MAX_ROTATABLE_BONDS = 15
    ROCS_THRESHOLD = 1.2


def load_datasets(
    actives_csv: str,
    decoys_csv: str,
    references_sdf: str,
) -> Tuple[List[str], List[str], List[Chem.Mol], List[str]]:
    """Load actives, decoys, and reference ligands.

    Args:
        actives_csv: Path to actives CSV (expects 'SMILES' column).
        decoys_csv: Path to decoys CSV (expects 'Smiles' column).
        references_sdf: Path to reference ligands SDF file.

    Returns:
        Tuple of (actives_smiles, decoys_smiles, ref_mols, ref_smiles).

    Raises:
        FileNotFoundError: If any input file is missing.
        ValueError: If files cannot be parsed or are empty.
    """
    if not os.path.exists(actives_csv):
        raise FileNotFoundError(f"Actives file not found: {actives_csv}")

    actives_df = pd.read_csv(actives_csv)
    if 'SMILES' not in actives_df.columns:
        raise ValueError(
            f"Actives CSV missing 'SMILES' column. "
            f"Found: {actives_df.columns.tolist()}"
        )
    actives_smiles = actives_df['SMILES'].tolist()

    if not os.path.exists(decoys_csv):
        raise FileNotFoundError(f"Decoys file not found: {decoys_csv}")

    decoys_df = pd.read_csv(decoys_csv)
    if 'Smiles' not in decoys_df.columns:
        raise ValueError(
            f"Decoys CSV missing 'Smiles' column. "
            f"Found: {decoys_df.columns.tolist()}"
        )
    decoys_smiles = decoys_df['Smiles'].tolist()

    if not os.path.exists(references_sdf):
        raise FileNotFoundError(f"References file not found: {references_sdf}")

    ref_suppl = Chem.SDMolSupplier(str(references_sdf), removeHs=False)
    ref_mols = [mol for mol in ref_suppl if mol is not None]
    ref_smiles = [Chem.MolToSmiles(mol) for mol in ref_mols]

    if not ref_mols:
        raise ValueError(f"No valid molecules in reference file: {references_sdf}")

    return actives_smiles, decoys_smiles, ref_mols, ref_smiles


def initialize_scorer(
    references_sdf: str,
    max_conformers: Optional[int] = None,
    max_isomers: Optional[int] = None,
    max_heavy_atoms: Optional[int] = None,
    max_rotatable_bonds: Optional[int] = None,
    num_threads: int = 1,
    n_jobs: int = -1,
) -> RDKitROCSScorer:
    """Initialize RDKit ROCS scorer.

    Args:
        references_sdf: Path to reference ligands.
        max_conformers: Maximum conformers per molecule.
        max_isomers: Maximum stereoisomers to enumerate.
        max_heavy_atoms: Maximum heavy atoms allowed.
        max_rotatable_bonds: Maximum rotatable bonds allowed.
        num_threads: Number of threads for RDKit conformer generation.
            1 = single-threaded (default, safe for all scenarios).
            0 = use all CPU cores (faster but may conflict with n_jobs>1).
            Set to 0 for maximum speed when using n_jobs=1.
        n_jobs: Number of parallel jobs for scoring (-1 = all CPUs).

    Returns:
        Configured RDKitROCSScorer instance.
    """
    if max_conformers is None:
        max_conformers = MAX_CONFORMERS
    if max_isomers is None:
        max_isomers = MAX_ISOMERS
    if max_heavy_atoms is None:
        max_heavy_atoms = MAX_HEAVY_ATOMS
    if max_rotatable_bonds is None:
        max_rotatable_bonds = MAX_ROTATABLE_BONDS

    scorer = RDKitROCSScorer(
        conformer_generator=RDKitConformerGenerator(
            max_conformers=max_conformers,
            max_isomers=max_isomers,
            max_heavy_atoms=max_heavy_atoms,
            max_rotatable_bonds=max_rotatable_bonds,
            num_threads=num_threads,
            show_progress=False,
        ),
        references=str(references_sdf),
        score_type='TanimotoCombo',
        use_colors=True,
        show_progress=False,
        n_jobs=n_jobs,
    )

    return scorer


def score_molecules(
    scorer: RDKitROCSScorer,
    actives_smiles: List[str],
    decoys_smiles: List[str],
    ref_mols: List[Chem.Mol],
) -> Dict[str, np.ndarray]:
    """Score all molecules using ROCS.

    Args:
        scorer: Configured RDKitROCSScorer.
        actives_smiles: List of active SMILES.
        decoys_smiles: List of decoy SMILES.
        ref_mols: List of reference molecules.

    Returns:
        Dictionary with keys: 'actives_scores', 'decoys_scores', 'ref_scores',
            'actives_mols', 'decoys_mols'.
    """
    actives_mols = [
        mol for mol in [Chem.MolFromSmiles(smi) for smi in actives_smiles]
        if mol is not None
    ]
    decoys_mols = [
        mol for mol in [Chem.MolFromSmiles(smi) for smi in decoys_smiles]
        if mol is not None
    ]

    actives_scores = scorer.getScores(actives_mols).flatten()
    decoys_scores = scorer.getScores(decoys_mols).flatten()
    ref_scores = scorer.getScores(ref_mols).flatten()

    return {
        'actives_scores': actives_scores,
        'decoys_scores': decoys_scores,
        'ref_scores': ref_scores,
        'actives_mols': actives_mols,
        'decoys_mols': decoys_mols
    }


def perform_roc_analysis(
    actives_scores: np.ndarray,
    decoys_scores: np.ndarray,
) -> Dict[str, Any]:
    """Perform ROC analysis and find optimal threshold.

    Args:
        actives_scores: Scores for active molecules.
        decoys_scores: Scores for decoy molecules.

    Returns:
        Dictionary with ROC data and optimal threshold.
    """
    y_true = np.concatenate([
        np.ones(len(actives_scores)),
        np.zeros(len(decoys_scores))
    ])
    y_scores = np.concatenate([actives_scores, decoys_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    youdens_index = tpr - fpr
    optimal_idx = np.argmax(youdens_index)
    optimal_threshold = thresholds[optimal_idx]

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'roc_auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'optimal_tpr': tpr[optimal_idx],
        'optimal_fpr': fpr[optimal_idx],
        'optimal_idx': optimal_idx,
        'youdens_index': youdens_index,
        'precision': precision,
        'recall': recall,
        'pr_thresholds': pr_thresholds,
        'pr_auc': pr_auc,
        'y_true': y_true,
        'y_scores': y_scores
    }


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute classification metrics at a specific threshold."""
    predicted_actives = y_scores >= threshold
    tp = np.sum((y_true == 1) & predicted_actives)
    fp = np.sum((y_true == 0) & predicted_actives)
    tn = np.sum((y_true == 0) & ~predicted_actives)
    fn = np.sum((y_true == 1) & ~predicted_actives)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = (
        2 * precision_val * sensitivity / (precision_val + sensitivity)
        if (precision_val + sensitivity) > 0
        else 0
    )

    return {
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision_val,
        'f1': f1,
    }


def compare_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds_to_test: Optional[List[float]] = None,
    current_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Compare performance at different threshold values.

    Args:
        y_true: True labels (1=active, 0=decoy).
        y_scores: Predicted scores.
        thresholds_to_test: List of thresholds to evaluate.
        current_threshold: Current threshold value to ensure is included.

    Returns:
        DataFrame with threshold metrics.
    """
    if thresholds_to_test is None:
        thresholds_to_test = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    # Ensure current_threshold is included if provided
    if current_threshold is not None and current_threshold not in thresholds_to_test:
        thresholds_to_test = sorted(thresholds_to_test + [current_threshold])

    results = []
    for thresh in thresholds_to_test:
        metrics = compute_threshold_metrics(y_true, y_scores, thresh)

        results.append({
            'Threshold': thresh,
            'TPR (Sensitivity)': metrics['sensitivity'],
            'FPR': 1 - metrics['specificity'],
            'Precision': metrics['precision'],
            'F1-Score': metrics['f1'],
            'TP': metrics['tp'],
            'FP': metrics['fp'],
            'TN': metrics['tn'],
            'FN': metrics['fn'],
        })

    return pd.DataFrame(results)


def create_visualizations(
    actives_scores: np.ndarray,
    decoys_scores: np.ndarray,
    ref_scores: np.ndarray,
    roc_data: Dict,
    threshold_df: pd.DataFrame,
    current_threshold: float,
    show_plots: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create 3-panel visualization of threshold analysis.

    Args:
        actives_scores: Active molecule scores.
        decoys_scores: Decoy molecule scores.
        ref_scores: Reference ligand scores.
        roc_data: ROC analysis results.
        threshold_df: Threshold comparison DataFrame.
        current_threshold: Current threshold value.
        show_plots: Display plots interactively.
        save_path: Path to save figure (optional).

    Returns:
        Matplotlib figure object.
    """
    fig = plt.figure(figsize=(18, 5))

    ax1 = plt.subplot(1, 3, 1)
    bins = np.linspace(0, 2, 40)
    ax1.hist(
        actives_scores,
        bins=bins,
        alpha=0.6,
        label=f'Actives (n={len(actives_scores)})',
        color='blue',
        edgecolor='black',
        density=True
    )
    ax1.hist(
        decoys_scores,
        bins=bins,
        alpha=0.6,
        label=f'Decoys (n={len(decoys_scores)})',
        color='red',
        edgecolor='black',
        density=True
    )

    for score in ref_scores:
        ax1.axvline(score, color='green', linestyle='--', alpha=0.7, linewidth=2)
    if len(ref_scores) > 0:
        ax1.axvline(
            ref_scores[0],
            color='green',
            linestyle='--',
            alpha=0.7,
            linewidth=2,
            label=f'References (n={len(ref_scores)})'
        )

    ax1.axvline(
        current_threshold,
        color='purple',
        linestyle='-',
        linewidth=2.5,
        label=f'Current Threshold ({current_threshold})'
    )
    ax1.axvline(
        roc_data['optimal_threshold'],
        color='orange',
        linestyle=':',
        linewidth=2.5,
        label=f'Optimal Threshold ({roc_data["optimal_threshold"]:.2f})'
    )

    ax1.set_xlabel('TanimotoCombo Score', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax1.set_title('Score Distribution: Actives vs Decoys',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2)

    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(
        roc_data['fpr'],
        roc_data['tpr'],
        color='darkblue',
        lw=2,
        label=f'ROC (AUC = {roc_data["roc_auc"]:.3f})'
    )
    ax2.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
             label='Random')
    ax2.scatter(
        [roc_data['optimal_fpr']],
        [roc_data['optimal_tpr']],
        color='orange',
        s=150,
        zorder=5,
        label=f'Optimal ({roc_data["optimal_threshold"]:.2f})',
        marker='*',
        edgecolors='black'
    )

    current_idx = np.argmin(np.abs(roc_data['thresholds'] - current_threshold))
    ax2.scatter(
        [roc_data['fpr'][current_idx]],
        [roc_data['tpr'][current_idx]],
        color='purple',
        s=150,
        zorder=5,
        label=f'Current ({current_threshold})',
        marker='D',
        edgecolors='black'
    )

    ax2.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax2.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)

    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(
        threshold_df['Threshold'],
        threshold_df['TPR (Sensitivity)'],
        marker='o',
        linewidth=2,
        label='TPR (Sensitivity)',
        color='blue'
    )
    ax3.plot(
        threshold_df['Threshold'],
        threshold_df['FPR'],
        marker='s',
        linewidth=2,
        label='FPR',
        color='red'
    )
    ax3.plot(
        threshold_df['Threshold'],
        threshold_df['F1-Score'],
        marker='^',
        linewidth=2,
        label='F1-Score',
        color='green'
    )

    ax3.axvline(
        current_threshold,
        color='purple',
        linestyle='-',
        linewidth=2,
        alpha=0.5,
        label=f'Current ({current_threshold})'
    )
    ax3.axvline(
        roc_data['optimal_threshold'],
        color='orange',
        linestyle=':',
        linewidth=2,
        alpha=0.7,
        label=f'Optimal ({roc_data["optimal_threshold"]:.2f})'
    )

    ax3.set_xlabel('Threshold', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Metric Value', fontsize=11, fontweight='bold')
    ax3.set_title('Threshold Performance Metrics',
                  fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plots:
        plt.show()

    return fig


def generate_report(
    roc_data: Dict,
    threshold_df: pd.DataFrame,
    current_threshold: float,
    actives_scores: np.ndarray,
    decoys_scores: np.ndarray,
    ref_scores: np.ndarray
) -> str:
    """Generate comprehensive text report with recommendations.

    Args:
        roc_data: ROC analysis results.
        threshold_df: Threshold comparison DataFrame.
        current_threshold: Current threshold value.
        actives_scores: Active molecule scores.
        decoys_scores: Decoy molecule scores.
        ref_scores: Reference ligand scores.

    Returns:
        Formatted text report.
    """
    report = []
    report.append("=" * 80)
    report.append("RECOMMENDATIONS")
    report.append("=" * 80)

    quality = ('Excellent' if roc_data['roc_auc'] > 0.9 else
               'Good' if roc_data['roc_auc'] > 0.8 else 'Acceptable')
    report.append("\n1. ROC Analysis Results:")
    report.append(f"   - AUC: {roc_data['roc_auc']:.4f} ({quality})")
    report.append(
        f"   - Optimal threshold: {roc_data['optimal_threshold']:.3f} "
        f"(Youden's Index)"
    )
    report.append(
        f"   - At optimal: TPR={roc_data['optimal_tpr']:.3f}, "
        f"FPR={roc_data['optimal_fpr']:.3f}"
    )

    report.append("\n2. Current Threshold Analysis:")
    current_filtered = threshold_df[
        threshold_df['Threshold'] == current_threshold
    ]
    if len(current_filtered) > 0:
        current_metrics = current_filtered.iloc[0]
        report.append(f"   - Current ROCS_THRESHOLD: {current_threshold}")
        report.append(
            f"   - Sensitivity (TPR): {current_metrics['TPR (Sensitivity)']:.3f}"
        )
        report.append(f"   - False Positive Rate: {current_metrics['FPR']:.3f}")
        report.append(f"   - Precision: {current_metrics['Precision']:.3f}")
        report.append(f"   - F1-Score: {current_metrics['F1-Score']:.3f}")
    else:
        # Fallback: compute metrics on the fly if threshold not in DataFrame
        y_true = np.concatenate([
            np.ones(len(actives_scores)),
            np.zeros(len(decoys_scores))
        ])
        metrics = compute_threshold_metrics(y_true, np.concatenate([
            actives_scores,
            decoys_scores
        ]), current_threshold)
        report.append(f"   - Current ROCS_THRESHOLD: {current_threshold}")
        report.append(f"   - Sensitivity (TPR): {metrics['sensitivity']:.3f}")
        report.append(f"   - False Positive Rate: {1 - metrics['specificity']:.3f}")
        report.append(f"   - Precision: {metrics['precision']:.3f}")
        report.append(f"   - F1-Score: {metrics['f1']:.3f}")

    report.append("\n3. Recommendation:")
    threshold_diff = abs(current_threshold - roc_data['optimal_threshold'])
    if threshold_diff < 0.1:
        report.append(
            f"   Current threshold ({current_threshold}) is near-optimal. "
            f"No change needed."
        )
    elif current_threshold > roc_data['optimal_threshold'] + 0.2:
        report.append(
            f"   Current threshold ({current_threshold}) is too conservative "
            f"(missing actives)."
        )
        report.append(
            f"   Consider lowering to {roc_data['optimal_threshold']:.2f} "
            f"for better recall."
        )
    elif current_threshold < roc_data['optimal_threshold'] - 0.2:
        report.append(
            f"   Current threshold ({current_threshold}) is too permissive "
            f"(including decoys)."
        )
        report.append(
            f"   Consider raising to {roc_data['optimal_threshold']:.2f} "
            f"for better precision."
        )
    else:
        report.append(
            f"   Current threshold ({current_threshold}) is reasonable."
        )
        report.append(
            f"   Optimal would be {roc_data['optimal_threshold']:.2f}, "
            f"but current value is acceptable."
        )

    report.append("\n4. Distribution Overlap:")
    overlap_min = max(decoys_scores.min(), actives_scores.min())
    overlap_max = min(decoys_scores.max(), actives_scores.max())
    if overlap_max > overlap_min:
        overlap_actives = np.sum(
            (actives_scores >= overlap_min) & (actives_scores <= overlap_max)
        )
        overlap_decoys = np.sum(
            (decoys_scores >= overlap_min) & (decoys_scores <= overlap_max)
        )
        report.append(f"   - Overlap region: {overlap_min:.3f} - {overlap_max:.3f}")
        report.append(
            f"   - Actives in overlap: {overlap_actives}/{len(actives_scores)} "
            f"({100*overlap_actives/len(actives_scores):.1f}%)"
        )
        report.append(
            f"   - Decoys in overlap: {overlap_decoys}/{len(decoys_scores)} "
            f"({100*overlap_decoys/len(decoys_scores):.1f}%)"
        )
    else:
        report.append(
            "   Complete separation achieved! "
            "No overlap between actives and decoys."
        )

    report.append("\n5. Reference Ligands (Self-Similarity):")
    for i, score in enumerate(ref_scores):
        status = "FAILED" if score == 0 else "OK"
        report.append(f"   {status} Reference {i+1}: {score:.3f}")
    report.append(
        f"   - Mean: {ref_scores.mean():.3f} "
        f"(expect high scores for good templates)"
    )

    report.append("\n" + "=" * 80)
    report.append("Analysis complete! Use these insights to optimize "
                  "config.py settings.")
    report.append("=" * 80)

    return "\n".join(report)


def save_results(
    output_dir: str,
    actives_scores: np.ndarray,
    decoys_scores: np.ndarray,
    ref_scores: np.ndarray,
    actives_smiles: List[str],
    decoys_smiles: List[str],
    ref_smiles: List[str],
    roc_data: Dict,
    threshold_df: pd.DataFrame,
    report_text: str
) -> None:
    """Save all analysis results to disk.

    Args:
        output_dir: Output directory path.
        actives_scores: Active molecule scores.
        decoys_scores: Decoy molecule scores.
        ref_scores: Reference ligand scores.
        actives_smiles: Active SMILES.
        decoys_smiles: Decoy SMILES.
        ref_smiles: Reference SMILES.
        roc_data: ROC analysis results.
        threshold_df: Threshold comparison DataFrame.
        report_text: Formatted text report.
    """
    output_path = Path(output_dir)

    scores_data = []
    for smi, score in zip(actives_smiles[:len(actives_scores)], actives_scores):
        scores_data.append({'SMILES': smi, 'Label': 'Active', 'Score': score})
    for smi, score in zip(decoys_smiles[:len(decoys_scores)], decoys_scores):
        scores_data.append({'SMILES': smi, 'Label': 'Decoy', 'Score': score})
    for smi, score in zip(ref_smiles, ref_scores):
        scores_data.append({'SMILES': smi, 'Label': 'Reference', 'Score': score})

    scores_df = pd.DataFrame(scores_data)
    scores_df.to_csv(output_path / 'molecule_scores.csv', index=False)

    roc_df = pd.DataFrame({
        'FPR': roc_data['fpr'],
        'TPR': roc_data['tpr'],
        'Threshold': roc_data['thresholds']
    })
    roc_df.to_csv(output_path / 'roc_analysis.csv', index=False)

    threshold_df.to_csv(output_path / 'threshold_metrics.csv', index=False)

    with open(output_path / 'analysis_summary.txt', 'w') as f:
        f.write(report_text)


def run_threshold_analysis(
    actives_csv: str,
    decoys_csv: str,
    references_sdf: Optional[str] = None,
    current_threshold: Optional[float] = None,
    output_dir: str = 'threshold_analysis_results',
    max_conformers: Optional[int] = None,
    max_isomers: Optional[int] = None,
    max_heavy_atoms: Optional[int] = None,
    max_rotatable_bonds: Optional[int] = None,
    num_threads: Optional[int] = None,
    n_jobs: int = -1,
    show_plots: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run the ROCS threshold determination workflow.

    Args:
        actives_csv: Path to actives CSV file.
        decoys_csv: Path to decoys CSV file.
        references_sdf: Path to reference ligands SDF (uses config default).
        current_threshold: Current ROCS threshold value (uses config default).
        output_dir: Directory for saving results.
        max_conformers: Max conformers per molecule (uses config default).
        max_isomers: Max stereoisomers to enumerate (uses config default).
        max_heavy_atoms: Max heavy atoms allowed (uses config default).
        max_rotatable_bonds: Max rotatable bonds allowed (uses config default).
        num_threads: CPU threads for conformer generation. None auto-detects
            (n_jobs=1 -> num_threads=1, otherwise 0). Use 0 for all cores, 1
            for single-threaded.
        n_jobs: Number of parallel scoring jobs (default=-1, use all CPUs).
        show_plots: Whether to display plots.
        verbose: Whether to print progress messages.

    Returns:
        Dictionary with analysis results.

    Example usage:
        run_threshold_analysis(actives_csv, decoys_csv)
        run_threshold_analysis(actives_csv, decoys_csv, n_jobs=1)
        run_threshold_analysis(actives_csv, decoys_csv, num_threads=0, n_jobs=1)
    """
    if references_sdf is None:
        if CONFIG_AVAILABLE:
            references_sdf = str(CCR2_SDF)
        else:
            raise ValueError(
                "references_sdf must be provided if config.py is not available"
            )

    if current_threshold is None:
        current_threshold = ROCS_THRESHOLD

    # Smart default: auto-detect num_threads based on n_jobs
    # This ensures that when users set n_jobs=1 (resource limit),
    # we also use num_threads=1 (respects their intent)
    if num_threads is None:
        num_threads = 1 if n_jobs == 1 else 0

    actives_smiles, decoys_smiles, ref_mols, ref_smiles = load_datasets(
        actives_csv, decoys_csv, references_sdf
    )

    scorer = initialize_scorer(
        references_sdf, max_conformers, max_isomers,
        max_heavy_atoms, max_rotatable_bonds, num_threads, n_jobs
    )

    score_data = score_molecules(
        scorer, actives_smiles, decoys_smiles, ref_mols
    )

    roc_data = perform_roc_analysis(
        score_data['actives_scores'], score_data['decoys_scores']
    )

    threshold_df = compare_thresholds(
        roc_data['y_true'], roc_data['y_scores'], current_threshold=current_threshold
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig = create_visualizations(
        score_data['actives_scores'],
        score_data['decoys_scores'],
        score_data['ref_scores'],
        roc_data,
        threshold_df,
        current_threshold,
        show_plots=show_plots,
        save_path=str(output_path / 'combined_figure.png')
    )

    report_text = generate_report(
        roc_data, threshold_df, current_threshold,
        score_data['actives_scores'],
        score_data['decoys_scores'],
        score_data['ref_scores']
    )

    if verbose:
        print(report_text)

    save_results(
        output_dir,
        score_data['actives_scores'],
        score_data['decoys_scores'],
        score_data['ref_scores'],
        actives_smiles,
        decoys_smiles,
        ref_smiles,
        roc_data,
        threshold_df,
        report_text
    )

    return {
        'optimal_threshold': roc_data['optimal_threshold'],
        'roc_auc': roc_data['roc_auc'],
        'recommendation': report_text,
        'threshold_df': threshold_df,
        'roc_data': roc_data,
        'scores': score_data,
        'figure': fig
    }
