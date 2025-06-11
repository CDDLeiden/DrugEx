#!/usr/bin/env python3
"""Simple batch runner for multiple query/reference pairs with comprehensive scoring

This script compares three ROCS scoring implementations:
1. CLI-based ROCS (using rocs binary)
2. API-based ROCS CPU (using OpenEye Python API)
3. API-based ROCS GPU (using OpenEye Python API with GPU acceleration)

Note: The 'reference' files are actually previous CLI outputs, not external ground truth.
The goal is to compare consistency between different ROCS implementations.
"""

import subprocess
import os

# Define query/reference pairs
pairs = [
    ("../tutorial/rocs/vrocs_data/trypsin/Trypsin_1.sq", "../tutorial/rocs/model1", "trypsin1"),
    ("../tutorial/rocs/vrocs_data/trypsin/Trypsin_2.sq", "../tutorial/rocs/model2", "trypsin2"), 
    ("../tutorial/rocs/vrocs_data/trypsin/Trypsin_3.sq", "../tutorial/rocs/model3", "trypsin3")
]

def run_scoring_method(query_file, name, method_args, method_name):
    """Run a specific scoring method and return success status"""
    print(f"Running {method_name} scoring for {name}...")
    cmd = ["python", "unified_rocs_scorer.py"] + method_args + [
        "--query-file", query_file, "--query-name", name
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error in {method_name} scoring for {name}")
        return False
    return True

def main():
    for query_file, ref_file, name in pairs:
        print(f"=== Processing {name} ===")
        
        # Define scoring methods to run
        scoring_methods = [
            (["--cli"], "CLI"),
            (["--api"], "API CPU"), 
            (["--api", "--gpu"], "API GPU")
        ]
        
        # Run all scoring methods
        success_count = 0
        for method_args, method_name in scoring_methods:
            if run_scoring_method(query_file, name, method_args, method_name):
                success_count += 1
        
        if success_count == 0:
            print(f"All scoring methods failed for {name}, skipping analysis")
            continue
        
        print(f"Successfully completed {success_count}/3 scoring methods for {name}")
        
        # Run analysis (will analyze whatever results are available)
        print(f"Running analysis for {name}...")
        result = subprocess.run([
            "python", "rocs_results_analyzer.py",
            "--reference-file", ref_file, "--query-name", name
        ])
        
        if result.returncode != 0:
            print(f"Error in analysis for {name}")
            continue
            
        print(f"Completed {name}")
        print("-" * 50)

if __name__ == "__main__":
    main() 