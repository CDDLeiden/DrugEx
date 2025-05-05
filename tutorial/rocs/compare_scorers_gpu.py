#!/usr/bin/env python3
"""
ROCS Scorers GPU Performance Comparison Script

This script focuses specifically on GPU performance benchmarking for both ROCS implementations:
1. ez_rocs.RocsScorer - The simplified implementation
2. fastrocs.OpenEyeScorer - The optimized implementation

Includes GPU-specific optimizations and memory management.
"""

import os
import time
import argparse
import numpy as np
from pathlib import Path
import gc
import platform
import psutil
import warnings

# Suppress warning messages that might interfere with output
warnings.filterwarnings('ignore')
os.environ["OE_SILENT"] = "true"

# Initialize the OpenEye memory pool before importing any other modules
# that might also try to initialize it
from openeye import oechem
# Set global flag to prevent other modules from initializing memory pool again
os.environ["OE_MEMORY_POOL_INITIALIZED"] = "true"
try:
    # Initialize memory pool only once
    oechem.OESetMemPoolMode(oechem.OEMemPoolMode_System)
    print("Memory pool initialized in benchmark script")
except Exception as e:
    print(f"Warning: Failed to set memory pool mode: {e}")

# Now import the scorer modules
from drugex.training.scorers.ez_rocs import RocsScorer
from drugex.training.scorers.fastrocs import OpenEyeScorer

# Sample SMILES for testing (using a consistent set for both implementations)
SAMPLE_SMILES = [
    'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F',
    'CC1=C(C=CC=C1)NC(=O)CN2C=C(N=N2)C3=CC=CC=C3',
    'CC1=CC(=NO1)NC(=O)C2=CC=C(C=C2)NC(=O)C3CCCCC3',
    'C1=CC=C(C=C1)CC2=NN=C(O2)CN3C=CC(=CC3=O)O',
    'CCOC(=O)C1=CN=C(S1)NC(=O)C2=CC=C(C=C2)F',
    'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCOCC3)C',
    'CC1=CC(=CC=C1)C(=O)NC2=CC=C(C=C2)C3=CSC(=N3)N',
    'CC1=CC(=NO1)NC(=O)C2=CC=C(C=C2)NC(=O)C3CCCCC3',
    'CC1=CC=CC=C1NC(=O)C2=CC=C(C=C2)S(=O)(=O)N',
    'C1=CC(=CC=C1C(=O)NC2=CC=C(C=C2)S(=O)(=O)N)F',
]

def generate_test_data(n_molecules=100):
    """Generate test data by replicating sample SMILES."""
    repeats = (n_molecules + len(SAMPLE_SMILES) - 1) // len(SAMPLE_SMILES)
    smiles_list = []
    for _ in range(repeats):
        smiles_list.extend(SAMPLE_SMILES)
    return smiles_list[:n_molecules]

def find_model_file():
    """Find the ROCS shape query model file."""
    # Try common locations for the model file
    possible_paths = [
        Path("model3-4_v1.sq"),
        Path("tutorial/rocs/model3-4_v1.sq"),
        Path("../model3-4_v1.sq"),
        Path("../../model3-4_v1.sq"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    print("Warning: Model file not found. Please specify the path manually.")
    return None

def get_gpu_info():
    """Attempt to get GPU information using nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used', '--format=csv,noheader'], 
                               stdout=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except:
        return "GPU information not available"

def run_gpu_benchmark(scorer_class, name, batch_size, n_molecules, n_runs=3, 
                      max_isomers=2, max_rot_bonds=8, max_heavy_atoms=30, max_conformers=10):
    """
    Run GPU benchmark for a specific scorer implementation with multiple runs for better statistics.
    
    Parameters
    ----------
    scorer_class : class
        The scorer class to test (RocsScorer or OpenEyeScorer)
    name : str
        Name of the implementation for reporting
    batch_size : int
        Size of batches for processing
    n_molecules : int
        Total number of molecules to test
    n_runs : int
        Number of runs to perform for better statistics
    max_isomers : int
        Maximum isomers to generate
    max_rot_bonds : int
        Maximum rotatable bonds to consider
    max_heavy_atoms : int
        Maximum heavy atoms to process
    max_conformers : int
        Maximum conformers to generate
    """
    print(f"\n{'=' * 60}")
    title = f" {name} GPU Performance Test "
    print(f"{title.center(60, '=')}")
    print(f"{'=' * 60}\n")
    
    print(f"Implementation : {name}")
    print(f"Mode          : GPU (if available)")
    print(f"Batch size    : {batch_size}")
    print(f"Test molecules: {n_molecules}")
    print(f"Number of runs: {n_runs}")
    print(f"Max isomers   : {max_isomers}")
    print(f"Max conformers: {max_conformers}")
    
    # Verify GPU availability
    try:
        from openeye import oefastrocs
        gpu_ready = oefastrocs.OEFastROCSIsGPUReady()
        print(f"FastROCS GPU Status: {'READY' if gpu_ready else 'NOT AVAILABLE'}")
    except Exception as e:
        print(f"Error checking GPU status: {e}")
        gpu_ready = False
    
    # Generate test data
    smiles_list = generate_test_data(n_molecules)
    print(f"Generated {len(smiles_list)} test molecules")
    
    # Find model file
    model_path = find_model_file()
    if not model_path:
        return None
    
    # Initialize scorer
    try:
        print(f"Initializing {name} scorer...")
        start_init = time.time()
        
        # Initialize with GPU mode
        if scorer_class == RocsScorer:
            # RocsScorer from ez_rocs.py
            scorer = RocsScorer(
                sq_model_path=model_path,
                use_gpu=True,  # Try to use GPU
                max_isomers=max_isomers,  
                max_rot_bonds=max_rot_bonds,
                max_heavy_atoms=max_heavy_atoms,
                max_conformers=max_conformers
            )
        else:
            # OpenEyeScorer from fastrocs.py
            scorer = OpenEyeScorer(
                sq_model_path=model_path,
                use_gpu=True,  # Try to use GPU
                max_isomers=max_isomers,
                max_rot_bonds=max_rot_bonds,
                max_heavy_atoms=max_heavy_atoms,
                max_conformers=max_conformers
            )
        
        init_time = time.time() - start_init
        print(f"Initialization time: {init_time:.2f} seconds")
        
        # Perform warm-up run (critical for GPU benchmarks)
        print("Running warm-up batch...")
        _ = scorer.getScores(SAMPLE_SMILES[:10])  # Use smaller batch for warmup
        time.sleep(2)  # Let GPU settle
        
        # Force garbage collection before the main test
        gc.collect()
        
        # Create batches for processing
        batches = []
        batch_count = (n_molecules + batch_size - 1) // batch_size
        for i in range(0, n_molecules, batch_size):
            end_idx = min(i + batch_size, n_molecules)
            batches.append(smiles_list[i:end_idx])
        
        # Run multiple times for better statistics
        all_run_times = []
        all_batch_times = []
        
        print(f"Running {n_runs} benchmark runs...")
        
        for run in range(n_runs):
            print(f"\nRun {run+1}/{n_runs}:")
            # Set up timings for this run
            batch_times = []
            start_time = time.time()
            
            for i, batch in enumerate(batches):
                # Force garbage collection before each batch for consistent measurement
                gc.collect()
                
                batch_start = time.time()
                
                try:
                    scores = scorer.getScores(batch)
                    batch_time = time.time() - batch_start
                    
                    batch_times.append(batch_time)
                    
                    valid_count = np.sum(scores > 0)
                    print(f"  Batch {i+1}/{batch_count}: {len(batch)} molecules, {valid_count} valid, {batch_time:.2f} seconds")
                except Exception as e:
                    print(f"  Error processing batch {i+1}: {e}")
                    continue
                
                # Add brief cooling delay between batches to minimize thermal throttling
                if i < batch_count - 1:
                    time.sleep(0.3)
            
            if not batch_times:
                print("  No batches completed successfully")
                continue
                
            run_time = time.time() - start_time
            molecules_per_second = n_molecules / run_time
            
            print(f"  Run {run+1} results:")
            print(f"  - Total time: {run_time:.2f} seconds")
            print(f"  - Processing rate: {molecules_per_second:.2f} molecules/second")
            print(f"  - Avg batch time: {np.mean(batch_times):.2f} seconds")
            
            all_run_times.append(run_time)
            all_batch_times.extend(batch_times)
            
            # Add cooling period between runs
            if run < n_runs - 1:
                print("  Cooling period before next run...")
                time.sleep(5)
        
        if not all_run_times:
            print("No successful runs")
            return None
            
        # Calculate aggregate statistics
        avg_run_time = np.mean(all_run_times)
        avg_molecules_per_second = n_molecules / avg_run_time
        
        print(f"\nAggregate Results ({n_runs} runs):")
        print(f"Average total time   : {avg_run_time:.2f} seconds")
        print(f"Average processing rate: {avg_molecules_per_second:.2f} molecules/second")
        print(f"Average batch time   : {np.mean(all_batch_times):.2f} seconds")
        print(f"Median batch time    : {np.median(all_batch_times):.2f} seconds")
        print(f"Min batch time       : {np.min(all_batch_times):.2f} seconds")
        print(f"Max batch time       : {np.max(all_batch_times):.2f} seconds")
        print(f"Batch time std dev   : {np.std(all_batch_times):.3f} seconds")
        
        # Return the results for comparison
        return {
            'implementation': name,
            'mode': 'GPU' if gpu_ready else 'CPU',
            'avg_total_time': avg_run_time,
            'avg_molecules_per_second': avg_molecules_per_second,
            'avg_batch_time': np.mean(all_batch_times),
            'median_batch_time': np.median(all_batch_times),
            'min_batch_time': np.min(all_batch_times),
            'max_batch_time': np.max(all_batch_times),
            'std_batch_time': np.std(all_batch_times),
            'n_runs': n_runs
        }
    
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Cleanup
        del scorer
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description='Compare ROCS scorer GPU implementations')
    parser.add_argument('--ez-only', action='store_true', help='Test only ez_rocs implementation')
    parser.add_argument('--fast-only', action='store_true', help='Test only fastrocs implementation')
    parser.add_argument('--batch-size', type=int, default=30, help='Batch size')
    parser.add_argument('--molecules', type=int, default=300, help='Number of test molecules')
    parser.add_argument('--runs', type=int, default=3, help='Number of benchmark runs per implementation')
    parser.add_argument('--model-path', type=str, help='Path to ROCS shape query model file')
    args = parser.parse_args()
    
    # Print system information
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # Get memory information
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
    
    # Print GPU information
    print("\nGPU Information:")
    print(get_gpu_info())
    
    # Override model path if specified
    if args.model_path:
        global find_model_file
        find_model_file = lambda: args.model_path
    
    # Determine which implementations to test
    test_ez = not args.fast_only
    test_fast = not args.ez_only
    
    # Check for GPU availability
    try:
        from openeye import oefastrocs
        gpu_available = oefastrocs.OEFastROCSIsGPUReady()
        if not gpu_available:
            print("\nError: GPU not available for FastROCS. Cannot run GPU benchmarks.")
            return
        print(f"\nFastROCS GPU status: {'Ready' if gpu_available else 'Not available'}")
    except ImportError:
        print("\nError: FastROCS not available in this environment.")
        return
    
    results = {}
    
    # Test ez_rocs implementation
    if test_ez:
        results['ez_gpu'] = run_gpu_benchmark(
            RocsScorer, "ez_rocs.RocsScorer", 
            args.batch_size, args.molecules, args.runs
        )
    
    # Test fastrocs implementation
    if test_fast:
        results['fast_gpu'] = run_gpu_benchmark(
            OpenEyeScorer, "fastrocs.OpenEyeScorer", 
            args.batch_size, args.molecules, args.runs
        )
    
    # Compare results
    print("\n" + "=" * 60)
    print(" GPU Performance Comparison Summary ".center(60, "="))
    print("=" * 60 + "\n")
    
    valid_results = {k: v for k, v in results.items() if v}
    if not valid_results:
        print("No valid results to compare")
        return
    
    print(f"{'Implementation':<20} {'Molecules/sec':<15} {'Avg batch (s)':<15} {'Runs':<5}")
    print("-" * 60)
    
    for name, result in valid_results.items():
        print(f"{result['implementation']:<20} {result['avg_molecules_per_second']:<15.2f} {result['avg_batch_time']:<15.2f} {result['n_runs']}")
    
    # Compare implementations
    if 'ez_gpu' in valid_results and 'fast_gpu' in valid_results:
        speedup = valid_results['fast_gpu']['avg_molecules_per_second'] / valid_results['ez_gpu']['avg_molecules_per_second']
        print(f"\nGPU Speedup: fastrocs is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than ez_rocs")
        
        # Detailed comparison
        print("\nDetailed Metrics:")
        metrics = [
            ('Average batch time', 'avg_batch_time', 's'),
            ('Median batch time', 'median_batch_time', 's'),
            ('Min batch time', 'min_batch_time', 's'),
            ('Max batch time', 'max_batch_time', 's'),
            ('Batch time std dev', 'std_batch_time', 's'),
        ]
        
        for metric_name, metric_key, unit in metrics:
            ez_val = valid_results['ez_gpu'][metric_key]
            fast_val = valid_results['fast_gpu'][metric_key]
            ratio = fast_val / ez_val if ez_val > 0 else 0
            comparison = "lower" if metric_key in ['avg_batch_time', 'median_batch_time', 'max_batch_time', 'std_batch_time'] and ratio < 1 else "higher"
            print(f"{metric_name}: ez_rocs={ez_val:.3f}{unit}, fastrocs={fast_val:.3f}{unit} ({abs(1-ratio):.1%} {comparison})")

if __name__ == "__main__":
    main() 