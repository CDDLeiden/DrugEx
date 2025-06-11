#!/usr/bin/env python3
"""
Unified ROCS Scoring and Comparison Tool

This script combines the functionality of automate_rocs_compare.py and score_trypsin_with_rocs.py
into a single, comprehensive tool for ROCS-based molecular scoring and comparison.

Features:
- Score molecules using ROCS CLI or API
- Compare CLI vs API scoring methods
- Flexible file selection (ligands, decoys, or both)
- GPU/CPU mode selection for API
- Comprehensive comparison analysis with correlation metrics
- Dynamic output filename generation
- Robust error handling and validation

Example usage:

# Score with CLI only
python unified_rocs_scorer.py --cli

# Score with API using GPU
python unified_rocs_scorer.py --api --gpu

# Run both CLI and API, then compare
python unified_rocs_scorer.py --both

# Process only ligands with verbose output
python unified_rocs_scorer.py --ligands-only --verbose

# Skip comparison step
python unified_rocs_scorer.py --both --skip-comparison

"""

import os
import argparse
import pandas as pd
import sys
from cli_base_rocs import ROCSCLIScorer
from api_base_rocs import ROCSAPIScorer, ROCSAPIError


def get_file_label(filename):
    """Determine if file contains ligands (1) or decoys (0)"""
    if 'ligands' in filename.lower():
        return 1
    elif 'decoys' in filename.lower():
        return 0
    else:
        return 0  # default to inactive


def parse_arguments():
    """Parse command line arguments for unified ROCS scoring and comparison"""
    parser = argparse.ArgumentParser(
        description='Unified ROCS scoring and comparison tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --cli                    # CLI scoring only
  %(prog)s --api --gpu              # API scoring with GPU
  %(prog)s --both                   # Both CLI and API (default)
  %(prog)s --ligands-only           # Process ligands only
  %(prog)s --both --skip-comparison # Skip comparison analysis
        """
    )
    
    # Scoring method selection
    scoring_group = parser.add_mutually_exclusive_group()
    scoring_group.add_argument('--cli', action='store_true', 
                              help='Use CLI-based scorer only')
    scoring_group.add_argument('--api', action='store_true', 
                              help='Use API-based scorer only')
    scoring_group.add_argument('--both', action='store_true', 
                              help='Run both CLI and API scoring (default)')
    
    # API-specific options
    parser.add_argument('--gpu', action='store_true', 
                       help='Use GPU acceleration (API scorer only)')
    
    # File selection options
    file_group = parser.add_mutually_exclusive_group()
    file_group.add_argument('--ligands-only', action='store_true', 
                           help='Process only ligands file')
    file_group.add_argument('--decoys-only', action='store_true', 
                           help='Process only decoys file')
    
    # Output and behavior options
    parser.add_argument('--skip-comparison', action='store_true', 
                       help='Skip the comparison step')
    parser.add_argument('--verbose', action='store_true', 
                       help='Print detailed information')
    parser.add_argument('--output-dir', default='results', 
                       help='Output directory (default: results)')
    parser.add_argument('--base-name', default='trypsin_rocs_scores', 
                       help='Base name for output files (default: trypsin_rocs_scores)')
    
    # Query file options
    parser.add_argument('--query-file', default='../tutorial/rocs/vrocs_data/trypsin/Trypsin_1.sq',
                       help='Path to query file (default: Trypsin_1.sq)')
    parser.add_argument('--query-name', default='trypsin1',
                       help='Query name for output organization (default: trypsin1)')
    
    return parser.parse_args()


def generate_output_filename(base_name, use_api=False, use_gpu=False, output_dir='results', query_name='trypsin1'):
    """Generate a dynamic output filename based on scorer type and mode."""
    # Remove .tsv extension if present in base_name
    if base_name.endswith('.tsv'):
        base_name = base_name[:-4]
    
    # Generate tag based on scorer and mode
    if use_api:
        if use_gpu:
            tag = "api_gpu"
        else:
            tag = "api_cpu"
    else:
        tag = "cli"
    
    filename = f"{query_name}_rocs_scores_{tag}.tsv"
    return os.path.join(output_dir, filename)


def validate_input_files(query_file, molecule_files):
    """Validate that all required input files exist"""
    missing_files = []
    
    if not os.path.exists(query_file):
        missing_files.append(query_file)
    
    for mol_file in molecule_files:
        if not os.path.exists(mol_file):
            missing_files.append(mol_file)
    
    if missing_files:
        print(f"Error: Missing input files:")
        for file in missing_files:
            print(f"  - {file}")
        sys.exit(1)


def score_molecules(query_file, molecule_files, output_file, use_api=False, use_gpu=False, verbose=False):
    """Score molecules with ROCS and save directly to output file."""
    # Common parameters for both scorers - ENSURE EXACT MATCH
    common_params = {
        "query_files": query_file,
        "score_type": "TanimotoCombo",
        "max_conformers": 1,  # Consistent with automate_rocs_compare.py
        "optimize": True,
        "color_optimize": True,
        "color_force_field": "ImplicitMillsDean",
        "shape_only": False,
    }
    
    # Initialize ROCS scorer based on parameters
    if use_api:
        mode_desc = "GPU acceleration" if use_gpu else "CPU mode"
        if verbose:
            print(f"Using API-based ROCS scorer with {mode_desc}...")
        # Add API-specific parameters
        api_params = common_params.copy()
        api_params["use_gpu"] = use_gpu
        scorer = ROCSAPIScorer(**api_params)
        
        # For API scorer, collect all results first, then apply global ranking
        all_results = []
        for i, mol_file in enumerate(molecule_files):
            if verbose:
                print(f"Scoring {os.path.basename(mol_file)} ({i+1}/{len(molecule_files)})...")
            # Get results without writing to file yet
            file_label = get_file_label(mol_file)
            file_results = scorer.getScores(mol_file, output_file=None, append=False, file_label=file_label)
            if file_results:
                all_results.extend(file_results)
        
        # Apply global ranking across all results
        if all_results:
            all_results.sort(key=lambda x: x['TanimotoCombo'], reverse=True)
            for i, result in enumerate(all_results):
                result['Rank'] = i + 1
            
            # Write all results to output file at once with proper global ranking
            scorer._format_tsv_output(all_results, output_file, append=False, file_label=0)
            if verbose:
                print(f"Applied global ranking to {len(all_results)} molecules")
    else:
        if verbose:
            print("Using CLI-based ROCS scorer...")
        scorer = ROCSCLIScorer(**common_params)
        
        # For CLI scorer, also collect all results and apply global ranking
        if verbose:
            print(f"Processing {len(molecule_files)} file(s)...")
        
        all_files_data = []
        for i, mol_file in enumerate(molecule_files):
            if verbose:
                print(f"Scoring {os.path.basename(mol_file)} ({i+1}/{len(molecule_files)})...")
            
            # Create a temporary file for each ROCS run
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.tsv', delete=False) as tmp:
                temp_output = tmp.name
            
            # Run ROCS for this file with file label
            file_label = get_file_label(mol_file)
            scorer.getScores(mol_file, output_file=temp_output, append=False, file_label=file_label)
            
            # Read the results from temporary file
            try:
                import pandas as pd
                df = pd.read_csv(temp_output, sep='\t')
                df.columns = df.columns.str.strip()
                if not df.empty:
                    all_files_data.append(df)
            except Exception as e:
                print(f"Warning: Failed to read temporary results for {mol_file}: {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_output):
                    os.remove(temp_output)
        
        # Combine all results and apply global ranking
        if all_files_data:
            import pandas as pd
            combined_df = pd.concat(all_files_data, ignore_index=True)
            
            # Apply global ranking
            if 'TanimotoCombo' in combined_df.columns:
                combined_df = combined_df.sort_values(by='TanimotoCombo', ascending=False).reset_index(drop=True)
                combined_df['Rank'] = combined_df.index + 1
                
                # Save to final output file
                combined_df.to_csv(output_file, sep='\t', index=False)
                if verbose:
                    print(f"Applied global ranking to {len(combined_df)} molecules")
            else:
                print("Warning: TanimotoCombo column not found in CLI results")
        else:
            print("Warning: No results to combine from CLI scorer")


def sort_tsv_by_tanimoto(tsv_path, verbose=False):
    """Sort TSV file by TanimotoCombo score in descending order"""
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        # Remove any leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()
        
        if 'TanimotoCombo' in df.columns:
            df_sorted = df.sort_values(by='TanimotoCombo', ascending=False)
            df_sorted.to_csv(tsv_path, sep='\t', index=False)
            if verbose:
                print(f"Sorted {len(df_sorted)} molecules by TanimotoCombo score")
        else:
            print('[WARNING] TanimotoCombo column not found; skipping sort.')
    except Exception as e:
        print(f"[WARNING] Failed to sort results: {e}")


def compare_results(cli_output, api_output, output_dir, verbose=False):
    """Compare CLI and API results and generate correlation analysis"""
    try:
        if verbose:
            print("=== Comparing Results ===")
        
        # Read results
        cli_df = pd.read_csv(cli_output, sep="\t")
        api_df = pd.read_csv(api_output, sep="\t")

        # Strip whitespace from column names
        cli_df.columns = cli_df.columns.str.strip()
        api_df.columns = api_df.columns.str.strip()

        # Validate required columns
        if "TanimotoCombo" not in cli_df.columns:
            print("Error: 'TanimotoCombo' column not found in CLI output. Columns are:", list(cli_df.columns))
            return False
        if "TanimotoCombo" not in api_df.columns:
            print("Error: 'TanimotoCombo' column not found in API output. Columns are:", list(api_df.columns))
            return False

        # Extract base molecule IDs (remove conformer suffixes like _172)
        cli_df['base_id'] = cli_df['Name'].astype(str).str.split('_').str[0]
        api_df['base_id'] = api_df['Name'].astype(str).str.split('_').str[0]

        # Sort and rank - remove existing Rank columns first to avoid conflicts
        if "Rank" in cli_df.columns:
            cli_df = cli_df.drop("Rank", axis=1)
        if "Rank" in api_df.columns:
            api_df = api_df.drop("Rank", axis=1)
            
        cli_df = cli_df.sort_values(by="TanimotoCombo", ascending=False).reset_index(drop=True)
        api_df = api_df.sort_values(by="TanimotoCombo", ascending=False).reset_index(drop=True)
        cli_df["Rank_CLI"] = cli_df.index + 1
        api_df["Rank_API"] = api_df.index + 1

        # Merge and compare using base_id instead of full Name to handle conformer differences
        cli_merge = cli_df[["Name", "base_id", "TanimotoCombo", "Rank_CLI"]].copy()
        api_merge = api_df[["Name", "base_id", "TanimotoCombo", "Rank_API"]].copy()
        
        merged = pd.merge(cli_merge, api_merge, on="base_id", suffixes=("_CLI", "_API"))
        
        # Check if we have any matching molecules
        if len(merged) == 0:
            print("Warning: No matching molecules found between CLI and API results.")
            print(f"CLI has {len(cli_df)} molecules, API has {len(api_df)} molecules")
            print("This may be due to different molecule processing between CLI and API scorers.")
            
            # Save summary information instead
            summary_data = {
                "CLI_molecules": len(cli_df),
                "API_molecules": len(api_df),
                "Matching_molecules": 0,
                "CLI_top_score": cli_df["TanimotoCombo"].max() if len(cli_df) > 0 else 0,
                "API_top_score": api_df["TanimotoCombo"].max() if len(api_df) > 0 else 0,
                "CLI_mean_score": cli_df["TanimotoCombo"].mean() if len(cli_df) > 0 else 0,
                "API_mean_score": api_df["TanimotoCombo"].mean() if len(api_df) > 0 else 0
            }
            
            comparison_file = os.path.join(output_dir, "cli_api_summary.tsv")
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_csv(comparison_file, sep="\t", index=False)
            
            if verbose:
                print(f"Summary comparison saved to: {comparison_file}")
            
            return True
        
        merged["Rank_Diff"] = (merged["Rank_CLI"] - merged["Rank_API"]).abs()
        
        # Save comparison results
        comparison_file = os.path.join(output_dir, "cli_api_comparison.tsv")
        merged.to_csv(comparison_file, sep="\t", index=False)

        # Generate and display analysis
        spearman_corr = merged["Rank_CLI"].corr(merged["Rank_API"], method="spearman")
        print(f"Spearman correlation: {spearman_corr:.4f}")
        print(f"Number of matching molecules: {len(merged)}")
        
        # Show top discrepancies
        top_discrepancies = merged.sort_values("Rank_Diff", ascending=False).head(10)
        print("Top 10 ranking discrepancies:")
        print(top_discrepancies[["base_id", "Name_CLI", "Name_API", "Rank_CLI", "Rank_API", "Rank_Diff"]].to_string(index=False))
        
        if verbose:
            print(f"Detailed comparison saved to: {comparison_file}")
        
        return True
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        return False


def main():
    """Main function for unified ROCS scoring and comparison"""
    # Parse arguments
    args = parse_arguments()
    
    # Determine which scorers to run
    run_cli = False
    run_api = False
    
    if args.both or (not args.api and not args.cli):
        # Default behavior or explicit --both
        run_cli = True
        run_api = True
    elif args.cli:
        run_cli = True
    elif args.api:
        run_api = True
    
    if args.verbose:
        print(f"Running: CLI={run_cli}, API={run_api}")

    # Define file paths
    query_file = args.query_file
    ligands_file = "../tutorial/rocs/vrocs_data/trypsin/trypsin_ligands_confs.oeb.gz"
    decoys_file = "../tutorial/rocs/vrocs_data/trypsin/trypsin_decoys_confs.oeb.gz"
    
    # Determine which files to process based on arguments
    files_to_process = []
    if args.ligands_only:
        files_to_process = [ligands_file]
        if args.verbose:
            print("Processing ligands only")
    elif args.decoys_only:
        files_to_process = [decoys_file]
        if args.verbose:
            print("Processing decoys only")
    else:
        # Default behavior - process both files
        files_to_process = [ligands_file, decoys_file]
        if args.verbose:
            print("Processing both ligands and decoys")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate input files
    validate_input_files(query_file, files_to_process)

    # Generate output file paths
    cli_output = None
    api_output = None
    
    if run_cli:
        cli_output = generate_output_filename(args.base_name, use_api=False, output_dir=args.output_dir, query_name=args.query_name)
    if run_api:
        api_output = generate_output_filename(args.base_name, use_api=True, use_gpu=args.gpu, output_dir=args.output_dir, query_name=args.query_name)

    # Run CLI scoring
    if run_cli:
        print("=== Running ROCS CLI ===")
        try:
            score_molecules(query_file, files_to_process, cli_output, 
                          use_api=False, use_gpu=False, verbose=args.verbose)
            # Sort CLI output by TanimotoCombo
            sort_tsv_by_tanimoto(cli_output, verbose=args.verbose)
            print(f"CLI results saved to: {cli_output}")
        except Exception as e:
            print(f"CLI scoring failed: {e}")
            run_cli = False

    # Run API scoring
    if run_api:
        print("=== Running ROCS API ===")
        try:
            score_molecules(query_file, files_to_process, api_output, 
                          use_api=True, use_gpu=args.gpu, verbose=args.verbose)
            print(f"API results saved to: {api_output}")
        except ROCSAPIError as e:
            print(f"ROCS API Error: {e}")
            print("Recommendation: Use --cli mode for reliable scoring")
            run_api = False
        except Exception as e:
            print(f"API scoring failed: {e}")
            run_api = False

    # Compare results if both were run successfully
    if not args.skip_comparison and run_cli and run_api:
        success = compare_results(cli_output, api_output, args.output_dir, verbose=args.verbose)
        if not success:
            print("Comparison failed, but individual scoring results are available")
    elif not args.skip_comparison:
        if args.verbose:
            print("Skipping comparison (need both CLI and API results)")

    print("\nScoring complete!")
    if run_cli and cli_output:
        print(f"CLI results: {cli_output}")
    if run_api and api_output:
        print(f"API results: {api_output}")


if __name__ == "__main__":
    main() 