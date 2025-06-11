#!/usr/bin/env python3
"""
ROCS Results Analyzer
Compares different ROCS scoring implementations (CLI vs API CPU vs API GPU)

Note: The 'reference' files are from VROCS (valid baseline for comparison).
This analyzer validates consistency between different ROCS implementations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse
from scipy.stats import spearmanr

class ROCSAnalyzer:
    """Simple ROCS results analyzer"""
    
    def __init__(self, reference_file='tutorial/rocs/model1', query_name='trypsin1'):
        self.reference_file = reference_file
        self.query_name = query_name
        self.output_dir = Path(f"results/analysis/{query_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load reference and output files"""
        print("Loading data files...")
        
        # File paths
        files = {
            'reference': self.reference_file,
            'cli': f'results/{self.query_name}_rocs_scores_cli.tsv',
            'api_cpu': f'results/{self.query_name}_rocs_scores_api_cpu.tsv',
            'api_gpu': f'results/{self.query_name}_rocs_scores_api_gpu.tsv'
        }
        
        data = {}
        for name, path in files.items():
            if os.path.exists(path):
                try:
                    # Handle CSV vs TSV format
                    if name == 'reference':
                        df = pd.read_csv(path)  # CSV format
                    else:
                        df = pd.read_csv(path, sep='\t')  # TSV format
                    df.columns = df.columns.str.strip()
                    # Extract base molecule ID (remove conformer suffix)
                    df['base_id'] = df['Name'].astype(str).str.split('_').str[0]
                    data[name] = df
                    print(f"✓ Loaded {name}: {len(df)} molecules")
                except Exception as e:
                    print(f"✗ Failed to load {name}: {e}")
            else:
                print(f"✗ File not found: {path}")
        
        return data
    
    def merge_data(self, data):
        """Merge data on common molecule IDs"""
        if 'reference' not in data:
            print("Error: Reference file required")
            return None
            
        ref_df = data['reference'].copy()
        
        # Deduplicate reference data
        ref_df = ref_df.drop_duplicates(subset=['base_id']).copy()
        
        # Create merged dataset - handle different column names
        if 'ROCS_TanimotoCombo' in ref_df.columns:
            merged = ref_df[['base_id', 'Rank', 'ROCS_TanimotoCombo']].copy()
            merged.columns = ['molecule_id', 'ref_rank', 'ref_score']
        elif 'TanimotoCombo' in ref_df.columns:
            merged = ref_df[['base_id', 'Rank', 'TanimotoCombo']].copy()
            merged.columns = ['molecule_id', 'ref_rank', 'ref_score']
        else:
            print("Error: No TanimotoCombo column found in reference data")
            return None
        
        # Use actual active column if available (check multiple possible column names)
        if 'active' in ref_df.columns:
            merged['is_active'] = ref_df['active'].values
        elif 'Active' in ref_df.columns:
            merged['is_active'] = ref_df['Active'].values
        else:
            # Fallback: top 10% as actives
            threshold = np.percentile(merged['ref_score'], 90)
            merged['is_active'] = (merged['ref_score'] >= threshold).astype(int)
        
        # Add method results (deduplicated) and use Active column if available
        for method in ['cli', 'api_cpu', 'api_gpu']:
            if method in data:
                method_df = data[method].drop_duplicates(subset=['base_id']).copy()
                
                # Use Active column from method data if reference doesn't have it
                if 'is_active' not in merged.columns and 'Active' in method_df.columns:
                    method_df_with_active = method_df[['base_id', 'Rank', 'TanimotoCombo', 'Active']].copy()
                    method_df_with_active.columns = ['molecule_id', f'{method}_rank', f'{method}_score', 'is_active']
                    merged = merged.merge(method_df_with_active, on='molecule_id', how='inner')
                else:
                    method_df = method_df[['base_id', 'Rank', 'TanimotoCombo']].copy()
                    method_df.columns = ['molecule_id', f'{method}_rank', f'{method}_score']
                    merged = merged.merge(method_df, on='molecule_id', how='inner')
        
        print(f"✓ Merged data: {len(merged)} common molecules")
        print(f"✓ Actives: {merged['is_active'].sum()}, Decoys: {(1-merged['is_active']).sum()}")
        
        return merged
    
    def calculate_correlations(self, merged_df):
        """Calculate ranking correlations"""
        methods = [col.replace('_rank', '') for col in merged_df.columns if col.endswith('_rank') and col != 'ref_rank']
        
        correlations = {}
        for method in methods:
            if f'{method}_rank' in merged_df.columns:
                corr, p_val = spearmanr(merged_df['ref_rank'], merged_df[f'{method}_rank'])
                correlations[method] = {'correlation': corr, 'p_value': p_val}
                
                # High correlation validation for CLI vs VROCS
                if corr > 0.999 and method == 'cli':
                    score_diff = (merged_df['ref_score'] - merged_df[f'{method}_score']).abs().mean()
                    rank_diff = (merged_df['ref_rank'] - merged_df[f'{method}_rank']).abs().mean()
                    
                    print(f"\n✓ CLI-VROCS Consistency Validation")
                    print(f"CLI correlation with VROCS reference: {corr:.6f}")
                    print(f"Score difference: {score_diff:.6f}")
                    print(f"Rank difference: {rank_diff:.6f}")
                    
                    if score_diff < 0.01 and rank_diff < 5:
                        print(f"✓ EXPECTED: CLI produces identical results to VROCS baseline")
                        print(f"This confirms CLI implementation correctness\n")
                    else:
                        print(f"Note: Minor differences expected due to implementation variations\n")
                
                # Only flag non-CLI methods with suspiciously high correlation
                elif corr > 0.999 and method != 'cli':
                    score_diff = (merged_df['ref_score'] - merged_df[f'{method}_score']).abs().mean()
                    rank_diff = (merged_df['ref_rank'] - merged_df[f'{method}_rank']).abs().mean()
                    
                    print(f"\n*** UNUSUAL: {method.upper()} shows identical correlation to VROCS ***")
                    print(f"Correlation: {corr:.6f}, Score diff: {score_diff:.6f}, Rank diff: {rank_diff:.6f}")
                    print(f"This may indicate {method} is using same algorithm as VROCS/CLI\n")
        
        return correlations
    
    def calculate_enrichment_factors(self, merged_df):
        """Calculate enrichment factors for top 1%, 2%, 5%, 10%"""
        methods = [col.replace('_rank', '') for col in merged_df.columns if col.endswith('_rank') and col != 'ref_rank']
        percentiles = [1, 2, 5, 10]
        
        enrichment_data = {}
        random_rate = merged_df['is_active'].mean()
        
        for method in methods:
            if f'{method}_rank' in merged_df.columns:
                enrichment_data[method] = {}
                for pct in percentiles:
                    n_top = max(1, int(len(merged_df) * pct / 100))
                    top_n_df = merged_df.nsmallest(n_top, f'{method}_rank')
                    top_n_rate = top_n_df['is_active'].mean()
                    enrichment = top_n_rate / random_rate if random_rate > 0 else 0
                    enrichment_data[method][pct] = enrichment
        
        return enrichment_data
    
    def calculate_roc_data(self, merged_df):
        """Calculate ROC curves and AUC"""
        methods = [col.replace('_score', '') for col in merged_df.columns if col.endswith('_score') and col != 'ref_score']
        
        roc_data = {}
        for method in methods:
            if f'{method}_score' in merged_df.columns:
                scores = merged_df[f'{method}_score']
                labels = merged_df['is_active']
                
                # Sort by descending scores
                sorted_indices = np.argsort(scores)[::-1]
                sorted_labels = labels.iloc[sorted_indices].values
                
                # Calculate TPR and FPR
                n_pos = np.sum(labels)
                n_neg = len(labels) - n_pos
                
                tpr_list = [0]
                fpr_list = [0]
                
                tp = 0
                fp = 0
                
                for i in range(len(sorted_labels)):
                    if sorted_labels[i] == 1:
                        tp += 1
                    else:
                        fp += 1
                    
                    tpr = tp / n_pos if n_pos > 0 else 0
                    fpr = fp / n_neg if n_neg > 0 else 0
                    
                    tpr_list.append(tpr)
                    fpr_list.append(fpr)
                
                # Add endpoint
                tpr_list.append(1)
                fpr_list.append(1)
                
                fpr_array = np.array(fpr_list)
                tpr_array = np.array(tpr_list)
                auc = np.trapezoid(tpr_array, fpr_array)
                
                roc_data[method] = {
                    'fpr': fpr_array,
                    'tpr': tpr_array,
                    'auc': auc
                }
        
        return roc_data
    
    def plot_roc_curves(self, roc_data):
        """Plot ROC curves"""
        plt.figure(figsize=(10, 8))
        
        colors = {'ref': 'black', 'cli': 'blue', 'api_cpu': 'green', 'api_gpu': 'red'}
        
        for method, data in roc_data.items():
            color = colors.get(method, 'gray')
            plt.plot(data['fpr'], data['tpr'], 
                    color=color, linewidth=2, 
                    label=f"{method.upper()} (AUC = {data['auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_correlations(self, correlations):
        """Plot correlation heatmap"""
        methods = list(correlations.keys())
        corr_values = [correlations[m]['correlation'] for m in methods]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(methods, corr_values, color=['blue', 'green', 'red'])
        plt.ylim([0, 1])
        plt.ylabel('Spearman Correlation with Reference')
        plt.title('Ranking Correlation vs Reference')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, corr_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ranking_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_rank_shifts(self, merged_df):
        """Plot rank shift distributions"""
        methods = [col.replace('_rank', '') for col in merged_df.columns if col.endswith('_rank') and col != 'ref_rank']
        
        fig, axes = plt.subplots(1, len(methods), figsize=(15, 5))
        if len(methods) == 1:
            axes = [axes]
            
        colors = ['blue', 'green', 'red']
        
        # Calculate global min/max for consistent x-axis
        all_shifts = []
        for method in methods:
            if f'{method}_rank' in merged_df.columns:
                rank_shift = merged_df[f'{method}_rank'] - merged_df['ref_rank']
                all_shifts.extend(rank_shift.values)
        
        x_min, x_max = min(all_shifts), max(all_shifts)
        x_range = max(abs(x_min), abs(x_max)) * 1.1  # Add 10% padding
        
        for i, method in enumerate(methods):
            if f'{method}_rank' in merged_df.columns:
                rank_shift = merged_df[f'{method}_rank'] - merged_df['ref_rank']
                
                axes[i].hist(rank_shift, bins=50, alpha=0.7, color=colors[i % len(colors)])
                axes[i].axvline(x=0, color='red', linestyle='--', label='No Shift')
                axes[i].axvline(x=10, color='orange', linestyle='--', alpha=0.7, label='±10 Ranks')
                axes[i].axvline(x=-10, color='orange', linestyle='--', alpha=0.7)
                axes[i].set_xlabel('Rank Shift (Method - Reference)')
                axes[i].set_ylabel('Count')
                axes[i].set_title(f'{method.upper()} vs Reference')
                axes[i].set_xlim(-x_range, x_range)
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'rank_shift_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_top_n_recovery(self, merged_df):
        """Plot top-N recovery rates"""
        methods = [col.replace('_rank', '') for col in merged_df.columns if col.endswith('_rank') and col != 'ref_rank']
        top_n_values = [1, 5, 10, 20, 50, 100]
        
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'green', 'red']
        markers = ['o', 's', '^']
        linestyles = ['-', '--', '-.']
        
        for i, method in enumerate(methods):
            if f'{method}_rank' in merged_df.columns:
                recovery_rates = []
                for n in top_n_values:
                    ref_top_n = set(merged_df[merged_df['ref_rank'] <= n]['molecule_id'])
                    method_top_n = set(merged_df[merged_df[f'{method}_rank'] <= n]['molecule_id'])
                    recovery = len(ref_top_n.intersection(method_top_n)) / len(ref_top_n) * 100
                    recovery_rates.append(recovery)
                
                plt.plot(top_n_values, recovery_rates, 
                        marker=markers[i % len(markers)], 
                        linestyle=linestyles[i % len(linestyles)],
                        linewidth=2, markersize=8,
                        color=colors[i % len(colors)],
                        label=f'{method.upper()}')
        
        plt.xlabel('Top N Compounds')
        plt.ylabel('Recovery Rate (%)')
        plt.title('Top-N Recovery Rate vs Reference')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_n_recovery.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_enrichment_factors(self, enrichment_data):
        """Plot enrichment factors"""
        percentiles = [1, 2, 5, 10]
        methods = list(enrichment_data.keys())
        
        x = np.arange(len(percentiles))
        width = 0.25
        colors = ['blue', 'green', 'red']
        
        plt.figure(figsize=(10, 6))
        
        for i, method in enumerate(methods):
            values = [enrichment_data[method][pct] for pct in percentiles]
            plt.bar(x + i*width, values, width, label=method.upper(), color=colors[i % len(colors)])
        
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Random (EF=1)')
        plt.xlabel('Top N%')
        plt.ylabel('Enrichment Factor')
        plt.title('Enrichment Factors by Method')
        plt.xticks(x + width, [f'{p}%' for p in percentiles])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'enrichment_factors.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_summary(self, correlations, roc_data, merged_df, enrichment_data):
        """Generate text summary"""
        summary_file = self.output_dir / 'analysis_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("ROCS IMPLEMENTATION COMPARISON\n")
            f.write("=" * 50 + "\n")
            f.write("Comparing CLI vs API CPU vs API GPU implementations\n")
            f.write("Note: 'Reference' data is from VROCS (valid baseline)\n\n")
            f.write(f"Total molecules analyzed: {len(merged_df)}\n")
            f.write(f"Actives: {merged_df['is_active'].sum()}\n")
            f.write(f"Decoys: {(1-merged_df['is_active']).sum()}\n")
            f.write(f"Active rate: {merged_df['is_active'].mean():.3f}\n\n")
            
            # Method Performance (AUC)
            f.write("Method Performance (AUC):\n")
            auc_ranking = sorted([(k, v['auc']) for k, v in roc_data.items()], 
                               key=lambda x: x[1], reverse=True)
            for i, (method, auc) in enumerate(auc_ranking, 1):
                f.write(f"{i}. {method.upper()}: {auc:.3f}\n")
            f.write("\n")
            
            # Correlations with VROCS baseline
            f.write("Ranking Correlation with VROCS Baseline:\n")
            for method, data in correlations.items():
                corr = data['correlation']
                p_val = data['p_value']
                if method == 'cli' and corr > 0.99:
                    f.write(f"- {method.upper()}: {corr:.3f} (p={p_val:.2e}) ✓ Expected consistency\n")
                else:
                    f.write(f"- {method.upper()}: {corr:.3f} (p={p_val:.2e})\n")
            f.write("\n")
            
            # Enrichment factors
            f.write("Enrichment Factors (EF > 1 = better than random):\n")
            for method in enrichment_data.keys():
                f.write(f"- {method.upper()}:\n")
                for pct in [1, 2, 5, 10]:
                    ef = enrichment_data[method][pct]
                    f.write(f"  Top {pct}%: {ef:.2f}\n")
                f.write("\n")
        
        print(f"✓ Summary saved to {summary_file}")
        
    def run_analysis(self):
        """Run complete analysis"""
        print("Starting ROCS Results Analysis...")
        print("=" * 40)
        
        # Load and merge data
        data = self.load_data()
        if not data:
            print("Error: No data loaded")
            return
            
        merged_df = self.merge_data(data)
        if merged_df is None:
            print("Error: Failed to merge data")
            return
            
        # Calculate metrics
        correlations = self.calculate_correlations(merged_df)
        roc_data = self.calculate_roc_data(merged_df)
        enrichment_data = self.calculate_enrichment_factors(merged_df)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.plot_roc_curves(roc_data)
        self.plot_correlations(correlations)
        self.plot_rank_shifts(merged_df)
        self.plot_top_n_recovery(merged_df)
        self.plot_enrichment_factors(enrichment_data)
        
        # Save detailed data
        merged_df.to_csv(self.output_dir / 'detailed_comparison.csv', index=False)
        
        # Generate summary
        self.generate_summary(correlations, roc_data, merged_df, enrichment_data)
        
        print(f"\n✓ Analysis complete! Results saved to {self.output_dir}")
        print("\nGenerated files:")
        for file in self.output_dir.glob('*'):
            print(f"  - {file.name}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ROCS Results Analyzer')
    parser.add_argument('--reference-file', default='tutorial/rocs/model1',
                       help='Path to reference file (default: tutorial/rocs/model1)')
    parser.add_argument('--query-name', default='trypsin1',
                       help='Query name for output organization (default: trypsin1)')
    return parser.parse_args()

def main():
    """Main execution"""
    args = parse_arguments()
    analyzer = ROCSAnalyzer(reference_file=args.reference_file, query_name=args.query_name)
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 