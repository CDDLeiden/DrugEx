#!/usr/bin/env python3
"""
ROCS Scorers Comparison Script

This script benchmarks the two ROCS scorer implementations:
1. ez_rocs.RocsScorer - A simplified implementation prioritizing clarity and ease of use
   - Suitable for basic usage and educational purposes
   - Works well for small to medium molecule sets
   - Lower memory footprint than more complex implementations

2. fastrocs.OpenEyeScorer - An optimized implementation focusing on performance
   - Highly optimized with advanced memory management
   - Multi-threading and process pool support for CPU parallelization
   - Efficient resource-aware batching and caching
   - Designed for production environments and large molecule libraries

Both implementations support:
- GPU acceleration via FastROCS
- CPU-based calculations via ROCS
- Isomer enumeration and conformer generation

This script provides a fair comparison between both implementations
in both CPU and GPU modes with consistent parameters.
"""

import os
import time
import argparse
import numpy as np
from pathlib import Path
import gc
import platform
from tabulate import tabulate  # Add tabulate for better formatted tables

# Import both implementations
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

def run_benchmark(scorer_class, name, use_gpu, batch_size, n_molecules, cpu_processes=None, 
                 max_isomers=2, max_rot_bonds=8, max_heavy_atoms=30, max_conformers=10):
    """
    Run benchmark for a specific scorer implementation.
    
    Parameters
    ----------
    scorer_class : class
        The scorer class to test (RocsScorer or OpenEyeScorer)
    name : str
        Name of the implementation for reporting
    use_gpu : bool
        Whether to use GPU
    batch_size : int
        Size of batches for processing
    n_molecules : int
        Total number of molecules to test
    cpu_processes : int, optional
        Number of CPU processes to use (None = auto)
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
    title = f" {name} {'GPU' if use_gpu else 'CPU'} Performance Test "
    print(f"{title.center(60, '=')}")
    print(f"{'=' * 60}\n")
    
    print(f"Implementation : {name}")
    print(f"Mode          : {'GPU' if use_gpu else 'CPU'}")
    print(f"Batch size    : {batch_size}")
    print(f"Test molecules: {n_molecules}")
    if not use_gpu and cpu_processes:
        print(f"CPU processes : {cpu_processes}")
    
    # For fastrocs CPU mode, adjust batch size to avoid memory issues
    adjusted_batch_size = batch_size
    if scorer_class == OpenEyeScorer and not use_gpu:
        # Smaller batches for CPU mode of OpenEyeScorer to prevent memory issues
        adjusted_batch_size = min(15, batch_size)
        if adjusted_batch_size != batch_size:
            print(f"Adjusted batch size to {adjusted_batch_size} for CPU mode stability")
    
    # Generate test data
    smiles_list = generate_test_data(n_molecules)
    print(f"Generated {len(smiles_list)} test molecules")
    
    # Find model file
    model_path = find_model_file()
    if not model_path:
        return None
    
    # Initialize scorer with consistent settings for both implementations
    try:
        print(f"Initializing {name} scorer...")
        start_init = time.time()
        
        # Set environment variables for OpenEye memory management
        if scorer_class == OpenEyeScorer and not use_gpu:
            # Configure OpenEye environment for better memory handling
            os.environ["OE_SILENT"] = "true"
            # Use system memory pools for better stability
            os.environ["OE_MEMORY_POOL_MODE"] = "system"
            # Set a sensible thread limit
            os.environ["OE_NUM_THREADS"] = str(cpu_processes if cpu_processes else 2)
        
        if scorer_class == RocsScorer:
            # RocsScorer from ez_rocs.py
            scorer = RocsScorer(
                sq_model_path=model_path,
                use_gpu=use_gpu,
                max_isomers=max_isomers,  
                max_rot_bonds=max_rot_bonds,
                max_heavy_atoms=max_heavy_atoms,
                max_conformers=max_conformers,
                cpu_processes=cpu_processes
            )
        else:
            # OpenEyeScorer from fastrocs.py - use more conservative settings for CPU mode
            if not use_gpu:
                # Conservative settings for CPU mode
                scorer = OpenEyeScorer(
                    sq_model_path=model_path,
                    use_gpu=False,
                    max_isomers=1,  # Reduced for CPU stability
                    max_rot_bonds=6,  # Reduced for CPU stability
                    max_heavy_atoms=25,  # Reduced for CPU stability
                    max_conformers=8,   # Reduced for CPU stability
                    cpu_processes=min(2, cpu_processes if cpu_processes else 2)  # Limit processes to avoid OOM
                )
            else:
                # Standard settings for GPU mode
                scorer = OpenEyeScorer(
                    sq_model_path=model_path,
                    use_gpu=True,
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
        
        # Add extra delay for system stabilization
        time.sleep(2)
        
        # Force garbage collection before the main test
        gc.collect()
        
        # Batch the scoring for better measurements
        batches = []
        batch_count = (n_molecules + adjusted_batch_size - 1) // adjusted_batch_size
        for i in range(0, n_molecules, adjusted_batch_size):
            end_idx = min(i + adjusted_batch_size, n_molecules)
            batches.append(smiles_list[i:end_idx])
        
        print(f"Running {batch_count} batches (batch size: {adjusted_batch_size})...")
        
        # Set up timings
        batch_times = []
        molecules_per_batch = []
        start_time = time.time()
        
        for i, batch in enumerate(batches):
            # Ensure memory is clean before each batch
            if i > 0 and not use_gpu:
                gc.collect()
                
            batch_start = time.time()
            
            try:
                scores = scorer.getScores(batch)
                batch_time = time.time() - batch_start
                
                batch_times.append(batch_time)
                molecules_per_batch.append(len(batch))
                
                valid_count = np.sum(scores > 0)
                print(f"Batch {i+1}/{batch_count}: {len(batch)} molecules, {valid_count} valid, {batch_time:.2f} seconds")
            except Exception as e:
                print(f"Error processing batch {i+1}: {e}")
                # Continue with next batch if one fails
                continue
            
            # Add brief delay between batches to minimize thermal throttling on GPU
            # and to allow memory cleanup on CPU
            if i < batch_count - 1:
                if use_gpu:
                    time.sleep(0.3)
                else:
                    time.sleep(0.5)
        
        if not batch_times:
            print("No batches completed successfully")
            return None
            
        total_time = time.time() - start_time
        molecules_per_second = n_molecules / total_time
        
        print(f"\nBenchmark Results:")
        print(f"Total time          : {total_time:.2f} seconds")
        print(f"Processing rate     : {molecules_per_second:.2f} molecules/second")
        print(f"Average batch time  : {np.mean(batch_times):.2f} seconds")
        print(f"Median batch time   : {np.median(batch_times):.2f} seconds")
        print(f"Max batch time      : {np.max(batch_times):.2f} seconds")
        
        # Return the results for comparison
        return {
            'implementation': name,
            'mode': 'GPU' if use_gpu else 'CPU',
            'total_time': total_time,
            'molecules_per_second': molecules_per_second,
            'batch_times': batch_times
        }
    
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Cleanup
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description='Compare ROCS scorer implementations')
    parser.add_argument('--ez-only', action='store_true', help='Test only ez_rocs implementation')
    parser.add_argument('--fast-only', action='store_true', help='Test only fastrocs implementation')
    parser.add_argument('--gpu-only', action='store_true', help='Test only GPU mode')
    parser.add_argument('--cpu-only', action='store_true', help='Test only CPU mode')
    parser.add_argument('--batch-size', type=int, default=30, help='Batch size')
    parser.add_argument('--molecules', type=int, default=300, help='Number of test molecules')
    parser.add_argument('--cpu-processes', type=int, help='Number of CPU processes (auto if not specified)')
    parser.add_argument('--model-path', type=str, help='Path to ROCS shape query model file')
    args = parser.parse_args()
    
    # Print system information
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # Set CPU process count for consistent comparison
    cpu_processes = args.cpu_processes if args.cpu_processes is not None else 2  # Reduced default for stability
    
    # Override model path if specified
    if args.model_path:
        global find_model_file
        find_model_file = lambda: args.model_path
    
    # Determine which implementations to test
    test_ez = not args.fast_only
    test_fast = not args.ez_only
    
    # Determine which modes to test
    test_gpu = not args.cpu_only
    test_cpu = not args.gpu_only
    
    # Check for GPU availability
    gpu_available = False
    try:
        from openeye import oefastrocs
        gpu_available = oefastrocs.OEFastROCSIsGPUReady()
        if test_gpu and not gpu_available:
            print("Warning: GPU not available for FastROCS. Testing CPU mode only.")
            test_gpu = False
    except ImportError:
        print("Warning: FastROCS not available in this environment.")
        return
    
    results = {}
    
    # Print what will be tested
    print("\nBenchmark Configuration:")
    print(f"Testing CPU Mode: {'Yes' if test_cpu else 'No'}")
    print(f"Testing GPU Mode: {'Yes' if test_gpu and gpu_available else 'No'}")
    print(f"Testing ez_rocs: {'Yes' if test_ez else 'No'}")
    print(f"Testing fastrocs: {'Yes' if test_fast else 'No'}")
    print(f"Molecules per test: {args.molecules}")
    print(f"Batch size: {args.batch_size}")
    print(f"CPU processes: {cpu_processes}")
    print("-" * 60)
    
    # Test ez_rocs implementation
    if test_ez:
        if test_gpu and gpu_available:
            results['ez_gpu'] = run_benchmark(
                RocsScorer, "ez_rocs.RocsScorer", True, 
                args.batch_size, args.molecules
            )
        
        if test_cpu:
            results['ez_cpu'] = run_benchmark(
                RocsScorer, "ez_rocs.RocsScorer", False,
                args.batch_size, args.molecules, cpu_processes
            )
    
    # Test fastrocs implementation
    if test_fast:
        if test_gpu and gpu_available:
            results['fast_gpu'] = run_benchmark(
                OpenEyeScorer, "fastrocs.OpenEyeScorer", True,
                args.batch_size, args.molecules
            )
        
        if test_cpu:
            results['fast_cpu'] = run_benchmark(
                OpenEyeScorer, "fastrocs.OpenEyeScorer", False,
                args.batch_size, args.molecules, cpu_processes
            )
    
    # Compare results
    print("\n" + "=" * 60)
    print(" Performance Comparison Summary ".center(60, "="))
    print("=" * 60 + "\n")
    
    valid_results = {k: v for k, v in results.items() if v}
    if not valid_results:
        print("No valid results to compare")
        return
    
    # Create a table for better visualization
    table_data = []
    for name, result in valid_results.items():
        table_data.append([
            result['implementation'],
            result['mode'],
            f"{result['molecules_per_second']:.2f}",
            f"{result['total_time']:.2f}"
        ])
    
    # Print table
    print(tabulate(
        table_data,
        headers=["Implementation", "Mode", "Molecules/second", "Total time (s)"],
        tablefmt="grid"
    ))
    
    # Compare implementations in the same mode
    print("\nSpeed Comparisons:")
    comparisons = []
    
    # CPU mode comparison - highlighted first to emphasize CPU comparison
    if 'ez_cpu' in valid_results and 'fast_cpu' in valid_results:
        speedup = valid_results['fast_cpu']['molecules_per_second'] / valid_results['ez_cpu']['molecules_per_second']
        comparisons.append([
            "CPU mode comparison",
            f"fastrocs vs ez_rocs",
            f"{speedup:.2f}x {'faster' if speedup > 1 else 'slower'}"
        ])
    
    # GPU mode comparison
    if 'ez_gpu' in valid_results and 'fast_gpu' in valid_results:
        speedup = valid_results['fast_gpu']['molecules_per_second'] / valid_results['ez_gpu']['molecules_per_second']
        comparisons.append([
            "GPU mode comparison",
            f"fastrocs vs ez_rocs",
            f"{speedup:.2f}x {'faster' if speedup > 1 else 'slower'}"
        ])
    
    # GPU vs CPU within same implementation
    if 'ez_gpu' in valid_results and 'ez_cpu' in valid_results:
        speedup = valid_results['ez_gpu']['molecules_per_second'] / valid_results['ez_cpu']['molecules_per_second']
        comparisons.append([
            "ez_rocs implementation",
            f"GPU vs CPU",
            f"{speedup:.2f}x {'faster' if speedup > 1 else 'slower'}"
        ])
    
    if 'fast_gpu' in valid_results and 'fast_cpu' in valid_results:
        speedup = valid_results['fast_gpu']['molecules_per_second'] / valid_results['fast_cpu']['molecules_per_second']
        comparisons.append([
            "fastrocs implementation",
            f"GPU vs CPU",
            f"{speedup:.2f}x {'faster' if speedup > 1 else 'slower'}"
        ])
    
    # Print comparison table
    print(tabulate(
        comparisons,
        headers=["Comparison", "Implementations", "Performance Difference"],
        tablefmt="grid"
    ))
    
    # Print recommendations based on results
    print("\nRecommendations:")
    
    # Check if CPU-only results exist
    cpu_only_results = {k: v for k, v in valid_results.items() if 'cpu' in k}
    if cpu_only_results and len(cpu_only_results) > 1:
        best_cpu = max(cpu_only_results.items(), key=lambda x: x[1]['molecules_per_second'])
        print(f"- For CPU-only environments: Use {best_cpu[1]['implementation']}")
    
    # Check if GPU results exist
    gpu_results = {k: v for k, v in valid_results.items() if 'gpu' in k}
    if gpu_results and len(gpu_results) > 1:
        best_gpu = max(gpu_results.items(), key=lambda x: x[1]['molecules_per_second'])
        print(f"- For GPU-enabled environments: Use {best_gpu[1]['implementation']}")
    
    # Overall recommendation
    if valid_results:
        best_overall = max(valid_results.items(), key=lambda x: x[1]['molecules_per_second'])
        print(f"- Best overall performance: {best_overall[1]['implementation']} in {best_overall[1]['mode']} mode")

if __name__ == "__main__":
    try:
        from tabulate import tabulate
    except ImportError:
        # Simple tabulate fallback if the package is not available
        def tabulate(data, headers, tablefmt=None):
            result = []
            row_format = "{:<25} {:<20} {:<25}"
            result.append(row_format.format(*headers))
            result.append("-" * 70)
            for row in data:
                result.append(row_format.format(*row))
            return "\n".join(result)
    
    main() 