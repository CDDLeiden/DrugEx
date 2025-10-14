#!/usr/bin/env python
"""
Test script for ThresholdEstimator.

This script demonstrates threshold computation for all available ROCS scorers
and validates the results.
"""

import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from drugex.training.scorers.threshold_estimator import (
    ThresholdEstimator,
    quick_threshold_recommendation,
)

# Check backend availability
try:
    from drugex.training.scorers.conformer_generators import (
        CDPKitConformerGenerator,
        CDPL_AVAILABLE,
    )
    from drugex.training.scorers.cdpkit_rocs import CDPKitROCSAggregateScorer
except ImportError:
    CDPL_AVAILABLE = False

try:
    from drugex.training.scorers.conformer_generators import (
        OmegaConformerGenerator,
        OE_AVAILABLE as OMEGA_AVAILABLE,
    )
    from drugex.training.scorers.cli_rocs import (
        CLIROCSScorer,
        OE_AVAILABLE as ROCS_AVAILABLE,
    )
except ImportError:
    OMEGA_AVAILABLE = False
    ROCS_AVAILABLE = False

try:
    from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
    from drugex.training.scorers.rdkit_rocs import RDKitAggregateScorer
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

print("=" * 80)
print("Threshold Estimator Test")
print("=" * 80)
print(f"CDPKit Available: {CDPL_AVAILABLE}")
print(f"OpenEye Available: {OMEGA_AVAILABLE and ROCS_AVAILABLE}")
print(f"RDKit Available: {RDKIT_AVAILABLE}")
print("=" * 80)

# Reference molecules
REFERENCE_SDF = project_root / "tutorial/rocs/rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf"

if not REFERENCE_SDF.exists():
    print(f"\n❌ ERROR: Reference SDF not found at {REFERENCE_SDF}")
    sys.exit(1)

# Extract reference SMILES
reference_smiles = []
try:
    suppl = Chem.SDMolSupplier(str(REFERENCE_SDF), removeHs=False)
    for mol in suppl:
        if mol is not None:
            try:
                smi = Chem.MolToSmiles(mol)
                reference_smiles.append(smi)
            except:
                pass
    print(f"\n✓ Loaded {len(reference_smiles)} reference molecules")
except Exception as e:
    print(f"\n❌ Error loading references: {e}")
    sys.exit(1)

# Test 1: Quick recommendations (no scoring)
print("\n" + "=" * 80)
print("TEST 1: Quick Threshold Recommendations (No Scoring)")
print("=" * 80)

quick_results = [
    ("OpenEye ROCS Aggregate", "ROCS_CCR2", False),
    ("OpenEye ROCS Supermol", "ROCS_supermol", True),
    ("CDPKit ROCS Aggregate", "CDPKit_ROCS_TanimotoCombo", False),
    ("CDPKit ROCS Supermol", "CDPKit_ROCS_Supermol_TanimotoCombo", True),
    ("RDKit ROCS Aggregate", "RDKit_Aggregate_TanimotoCombo", False),
    ("RDKit ROCS Supermol", "RDKit_Supermolecule_TanimotoCombo", True),
]

quick_df_rows = []
for label, scorer_key, is_supermol in quick_results:
    threshold = quick_threshold_recommendation(scorer_key, is_supermol)
    quick_df_rows.append({
        "Backend": label,
        "Quick Threshold": threshold,
        "Method": "Supermolecule" if is_supermol else "Aggregate",
    })

quick_df = pd.DataFrame(quick_df_rows)
print("\n" + quick_df.to_string(index=False))

# Test 2: Computed thresholds (with scoring)
print("\n" + "=" * 80)
print("TEST 2: Computed Adaptive Thresholds (With Scoring)")
print("=" * 80)

estimator = ThresholdEstimator(show_progress=True)
computed_results = []

# Test CDPKit if available
if CDPL_AVAILABLE:
    print("\n📊 CDPKit ROCS Aggregate:")
    try:
        conf_gen = CDPKitConformerGenerator(
            max_conformers=50,
            timeout=1800,
            min_rmsd=0.5,
            energy_window=15.0,
            show_progress=False
        )
        scorer = CDPKitROCSAggregateScorer(
            conformer_generator=conf_gen,
            reference_mols=[str(REFERENCE_SDF)],
            show_progress=False
        )
        
        result = estimator.compute_threshold(
            scorer=scorer,
            reference_smiles=reference_smiles,
            strategy="percentile_75"
        )
        
        if result["status"] == "success":
            computed_results.append({
                "Backend": "CDPKit Aggregate",
                "Status": "success",
                "Computed Threshold": result["threshold"],
                "Mean Score": result["statistics"]["mean"],
                "Median Score": result["statistics"]["median"],
                "Q75 Score": result["statistics"]["q75"],
                "Max Score": result["statistics"]["max"],
            })
        else:
            print(f"   ❌ {result['status']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

# Test OpenEye if available
if ROCS_AVAILABLE:
    print("\n📊 OpenEye ROCS:")
    try:
        conf_gen = OmegaConformerGenerator(
            max_conformers=50,
            show_progress=False
        )
        scorer = CLIROCSScorer(
            conformer_generator=conf_gen,
            query_files={"CCR2": str(REFERENCE_SDF)},
            score_type="TanimotoCombo",
            show_progress=False
        )
        
        result = estimator.compute_threshold(
            scorer=scorer,
            reference_smiles=reference_smiles,
            strategy="percentile_75"
        )
        
        if result["status"] == "success":
            computed_results.append({
                "Backend": "OpenEye ROCS",
                "Status": "success",
                "Computed Threshold": result["threshold"],
                "Mean Score": result["statistics"]["mean"],
                "Median Score": result["statistics"]["median"],
                "Q75 Score": result["statistics"]["q75"],
                "Max Score": result["statistics"]["max"],
            })
        else:
            print(f"   ❌ {result['status']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

# Test RDKit if available
if RDKIT_AVAILABLE:
    print("\n📊 RDKit ROCS Aggregate:")
    try:
        # Load reference molecules with conformers
        ref_mols = []
        suppl = Chem.SDMolSupplier(str(REFERENCE_SDF), removeHs=False)
        for mol in suppl:
            if mol is not None and mol.GetNumConformers() > 0:
                ref_mols.append(mol)
        
        if ref_mols:
            conf_gen = RDKitConformerGenerator(
                max_conformers=50,
                show_progress=False
            )
            scorer = RDKitAggregateScorer(
                conformer_generator=conf_gen,
                reference_mols=ref_mols,
                show_progress=False
            )
            
            result = estimator.compute_threshold(
                scorer=scorer,
                reference_smiles=reference_smiles,
                strategy="percentile_75"
            )
            
            if result["status"] == "success":
                computed_results.append({
                    "Backend": "RDKit Aggregate",
                    "Status": "success",
                    "Computed Threshold": result["threshold"],
                    "Mean Score": result["statistics"]["mean"],
                    "Median Score": result["statistics"]["median"],
                    "Q75 Score": result["statistics"]["q75"],
                    "Max Score": result["statistics"]["max"],
                })
            else:
                print(f"   ❌ {result['status']}")
        else:
            print(f"   ⚠️  No valid reference molecules with conformers")
    except Exception as e:
        print(f"   ❌ Error: {e}")

# Display computed results
if computed_results:
    print("\n" + "=" * 80)
    print("COMPUTED THRESHOLD SUMMARY")
    print("=" * 80)
    
    computed_df = pd.DataFrame(computed_results)
    print("\n" + computed_df.to_string(index=False))
    
    # Compare with quick estimates
    print("\n" + "=" * 80)
    print("COMPARISON: Quick vs Computed Thresholds")
    print("=" * 80)
    
    comparison_rows = []
    for _, row in computed_df.iterrows():
        backend = row["Backend"]
        computed = row["Computed Threshold"]
        
        # Find matching quick estimate
        is_supermol = "Supermol" in backend
        scorer_key = backend.replace(" Aggregate", "").replace(" Supermol", "")
        
        if "OpenEye" in backend:
            quick = quick_threshold_recommendation("ROCS", is_supermol)
        elif "CDPKit" in backend:
            quick = quick_threshold_recommendation("CDPKit_ROCS", is_supermol)
        elif "RDKit" in backend:
            quick = quick_threshold_recommendation("RDKit", is_supermol)
        else:
            quick = 0.35
        
        comparison_rows.append({
            "Backend": backend,
            "Quick Estimate": quick,
            "Computed (P75)": computed,
            "Difference": computed - quick,
            "% Difference": (computed - quick) / quick * 100,
        })
    
    comparison_df = pd.DataFrame(comparison_rows)
    print("\n" + comparison_df.to_string(index=False))
    
    print("\n💡 Interpretation:")
    print("   - Quick estimates are rule-of-thumb values")
    print("   - Computed values are based on actual reference scores")
    print("   - Use computed values for production")
    print("   - Use quick estimates for initial exploration")

else:
    print("\n⚠️  No computed results available")
    print("   Ensure at least one backend is available and configured correctly")

print("\n" + "=" * 80)
print("Test Complete")
print("=" * 80)

