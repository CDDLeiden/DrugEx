#!/usr/bin/env python
"""
CDPKit vs OpenEye ROCS Diagnostic Script

This script compares CDPKit and OpenEye ROCS scoring to identify why
CDPKit produces significantly lower scores than OpenEye for the same molecules.

It tests:
1. Conformer generation quality (CDPKit vs OpenEye Omega)
2. Shape scoring differences (CDPKit ROCS vs OpenEye ROCS)
3. Score distribution analysis
4. Recommended threshold adjustments
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from drugex.training.scorers.conformer_generators import (
    CDPKitConformerGenerator,
    CDPL_AVAILABLE,
)
from drugex.training.scorers.cdpkit_rocs import (
    CDPKitROCSAggregateScorer,
)

# Check OpenEye availability
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

print("CDPKit vs OpenEye ROCS Diagnostic Analysis")
print(f"CDPKit: {CDPL_AVAILABLE}, OpenEye Omega: {OMEGA_AVAILABLE}, OpenEye ROCS: {ROCS_AVAILABLE}")

# Test molecules - a mix of known actives and random molecules
TEST_SMILES = [
    "CCOc1ccc(C(=O)Nc2ccc(S(=O)(=O)NC3CCCCC3)cc2)cc1",  # Similar to CCR2 ligands
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen (simpler)
    "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",  # Diazepam
    "c1ccc(cc1)Cc2ccccc2",  # Bibenzyl (simple)
    "COc1ccc2nc(S(=O)(=O)Nc3ccccc3)sc2c1",  # Complex sulfonamide
]

TEST_NAMES = [
    "CCR2-like",
    "Ibuprofen",
    "Diazepam",
    "Bibenzyl",
    "Sulfonamide",
]

# Reference molecules path
REFERENCE_SDF = project_root / "tutorial/rocs/rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf"

if not REFERENCE_SDF.exists():
    print(f"\nERROR: Reference SDF not found at {REFERENCE_SDF}")
    sys.exit(1)

print(f"\nUsing reference molecules: {REFERENCE_SDF}")


def analyze_conformers(conf_file, label):
    """Analyze generated conformers"""
    if not os.path.exists(conf_file):
        print(f"  {label}: Conformer file not generated")
        return 0, 0

    try:
        suppl = Chem.SDMolSupplier(conf_file, removeHs=False)
        total_confs = 0
        mol_conf_counts = {}

        for mol in suppl:
            if mol is None:
                continue
            total_confs += 1
            try:
                name = mol.GetProp("_Name")
                mol_id = name.split("+")[0]
                mol_conf_counts[mol_id] = mol_conf_counts.get(mol_id, 0) + 1
            except:
                pass

        unique_mols = len(mol_conf_counts)
        avg_confs = total_confs / unique_mols if unique_mols > 0 else 0

        print(f"  {label}: {unique_mols} molecules, {total_confs} total conformers "
              f"(avg {avg_confs:.1f} confs/mol)")
        return unique_mols, total_confs
    except Exception as e:
        print(f"  {label}: Error reading conformers - {e}")
        return 0, 0


print("\nPHASE 1: Conformer Generation Comparison")

results = []

with tempfile.TemporaryDirectory() as tmpdir:
    # Test CDPKit conformer generation
    if CDPL_AVAILABLE:
        print("\nTesting CDPKit Conformer Generation...")
        cdpkit_gen = CDPKitConformerGenerator(
            max_conformers=100,
            timeout=1800,
            min_rmsd=0.5,  # CDPKit default for diversity
            energy_window=15.0,
            show_progress=True
        )
        cdpkit_conf_file = cdpkit_gen.genConformers(TEST_SMILES, tmpdir)
        cdpkit_mols, cdpkit_confs = analyze_conformers(cdpkit_conf_file, "CDPKit")
    else:
        print("\nCDPKit not available, skipping")
        cdpkit_conf_file = None

    # Test OpenEye Omega conformer generation
    if OMEGA_AVAILABLE:
        print("\nTesting OpenEye Omega Conformer Generation...")
        omega_gen = OmegaConformerGenerator(
            max_conformers=200,
            show_progress=True
        )
        omega_conf_file = omega_gen.genConformers(TEST_SMILES, tmpdir)
        omega_mols, omega_confs = analyze_conformers(omega_conf_file, "Omega")
    else:
        print("\n⚠️  OpenEye Omega not available, skipping")
        omega_conf_file = None

    print("\n" + "=" * 80)
    print("PHASE 2: Shape Scoring Comparison")
    print("=" * 80)

    # Test CDPKit ROCS scoring
    if CDPL_AVAILABLE and cdpkit_conf_file:
        print("\n🎯 Testing CDPKit ROCS Scoring...")
        cdpkit_gen_for_scoring = CDPKitConformerGenerator(
            max_conformers=100,
            timeout=1800,
            min_rmsd=0.5,  # CDPKit default for diversity
            energy_window=15.0,
            show_progress=False
        )
        cdpkit_scorer = CDPKitROCSAggregateScorer(
            conformer_generator=cdpkit_gen_for_scoring,
            reference_mols=[str(REFERENCE_SDF)],
            show_progress=True
        )
        cdpkit_scores = cdpkit_scorer.getScores(TEST_SMILES)

        print(f"\n  CDPKit Scores:")
        for i, (name, smi, score) in enumerate(zip(TEST_NAMES, TEST_SMILES, cdpkit_scores)):
            print(f"    {name:15s}: {score[0]:.4f}")
            results.append({
                "Molecule": name,
                "SMILES": smi,
                "Scorer": "CDPKit",
                "Score": float(score[0])
            })

    # Test OpenEye ROCS scoring
    if ROCS_AVAILABLE and omega_conf_file:
        print("\n🎯 Testing OpenEye ROCS Scoring...")
        omega_gen_for_scoring = OmegaConformerGenerator(
            max_conformers=200,
            show_progress=False
        )
        rocs_scorer = CLIROCSScorer(
            conformer_generator=omega_gen_for_scoring,
            query_files={"CCR2": str(REFERENCE_SDF)},
            score_type="TanimotoCombo",
            show_progress=True
        )
        rocs_scores = rocs_scorer.getScores(TEST_SMILES)

        print(f"\n  OpenEye ROCS Scores:")
        for i, (name, smi, score) in enumerate(zip(TEST_NAMES, TEST_SMILES, rocs_scores)):
            print(f"    {name:15s}: {score[0]:.4f}")
            results.append({
                "Molecule": name,
                "SMILES": smi,
                "Scorer": "OpenEye ROCS",
                "Score": float(score[0])
            })

    print("\n" + "=" * 80)
    print("PHASE 2B: Self-Scoring Test (Reference Molecules vs Themselves)")
    print("=" * 80)
    print("\nℹ️  This tests if each reference molecule scores when regenerated from SMILES.")
    print("   Expected: CDPKit=~2.0 (raw), OpenEye ROCS=~2.0 (raw TanimotoCombo)")
    print("   Note: Scores may be <2.0 because SMILES loses 3D info and conformers are regenerated.")

    # Extract SMILES from reference molecules
    ref_smiles = []
    ref_names = []
    try:
        suppl = Chem.SDMolSupplier(str(REFERENCE_SDF), removeHs=False)
        for idx, mol in enumerate(suppl):
            if mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(mol)
                # Get name or create unique one
                name = mol.GetProp("_Name") if mol.HasProp("_Name") and mol.GetProp("_Name").strip() else f"Ref_{idx}"
                ref_smiles.append(smi)
                ref_names.append(name)
            except Exception as e:
                print(f"Warning: Error processing molecule {idx}: {e}")
                pass
        print(f"\n✓ Loaded {len(ref_smiles)} reference molecules for self-scoring test")
    except Exception as e:
        print(f"\n❌ Error loading reference molecules: {e}")
        ref_smiles = []

    # Test CDPKit self-scoring
    if CDPL_AVAILABLE and ref_smiles:
        print("\n🎯 CDPKit Self-Scoring Test...")
        
        # Generate conformers and count them
        with tempfile.TemporaryDirectory() as self_tmpdir:
            cdpkit_gen_self = CDPKitConformerGenerator(
                max_conformers=100,
                timeout=1800,
                min_rmsd=0.5,  # CDPKit default for diversity
                energy_window=15.0,
                show_progress=False
            )
            cdpkit_self_conf_file = cdpkit_gen_self.genConformers(ref_smiles[:3], self_tmpdir)
            
            # Count conformers per molecule
            if os.path.exists(cdpkit_self_conf_file):
                suppl = Chem.SDMolSupplier(cdpkit_self_conf_file, removeHs=False)
                conf_counts = {}
                for mol in suppl:
                    if mol is None:
                        continue
                    try:
                        name = mol.GetProp("_Name")
                        mol_id = name.split("+")[0]
                        conf_counts[mol_id] = conf_counts.get(mol_id, 0) + 1
                    except:
                        pass
                
                if conf_counts:
                    print(f"\n  Conformers generated per molecule:")
                    for i, mol_id in enumerate(sorted(conf_counts.keys())):
                        mol_name = ref_names[i] if i < len(ref_names) else f"Ref_{i}"
                        print(f"    {mol_name:20s}: {conf_counts[mol_id]} conformers")
                else:
                    print(f"\n  ⚠️  No conformers generated by CDPKit!")
            else:
                print(f"\n  ⚠️  Conformer file not found!")
        
        cdpkit_self_scores = cdpkit_scorer.getScores(ref_smiles[:3])  # Test first 3

        print(f"\n  CDPKit Self-Scores (Expected ~2.0):")
        for name, smi, score in zip(ref_names[:3], ref_smiles[:3], cdpkit_self_scores):
            print(f"    {name:20s}: {score[0]:.6f}")
            results.append({
                "Molecule": f"{name} (self)",
                "SMILES": smi,
                "Scorer": "CDPKit (self)",
                "Score": float(score[0])
            })

    # Test OpenEye ROCS self-scoring
    if ROCS_AVAILABLE and ref_smiles:
        print("\n🎯 OpenEye ROCS Self-Scoring Test...")
        
        # Generate conformers and count them
        with tempfile.TemporaryDirectory() as self_tmpdir:
            omega_gen_self = OmegaConformerGenerator(
                max_conformers=100,
                show_progress=False
            )
            omega_self_conf_file = omega_gen_self.genConformers(ref_smiles[:3], self_tmpdir)
            
            # Count conformers per molecule
            if os.path.exists(omega_self_conf_file):
                suppl = Chem.SDMolSupplier(omega_self_conf_file, removeHs=False)
                conf_counts = {}
                for mol in suppl:
                    if mol is None:
                        continue
                    try:
                        name = mol.GetProp("_Name")
                        mol_id = name.split("+")[0]
                        conf_counts[mol_id] = conf_counts.get(mol_id, 0) + 1
                    except:
                        pass
                
                if conf_counts:
                    print(f"\n  Conformers generated per molecule:")
                    for i, mol_id in enumerate(sorted(conf_counts.keys())):
                        mol_name = ref_names[i] if i < len(ref_names) else f"Ref_{i}"
                        print(f"    {mol_name:20s}: {conf_counts[mol_id]} conformers")
                else:
                    print(f"\n  ⚠️  No conformers generated by Omega!")
            else:
                print(f"\n  ⚠️  Conformer file not found: {omega_self_conf_file}")
        
        rocs_self_scores = rocs_scorer.getScores(ref_smiles[:3])  # Test first 3

        print(f"\n  OpenEye ROCS Self-Scores (Expected ~2.0):")
        for name, smi, score in zip(ref_names[:3], ref_smiles[:3], rocs_self_scores):
            print(f"    {name:20s}: {score[0]:.6f}")
            results.append({
                "Molecule": f"{name} (self)",
                "SMILES": smi,
                "Scorer": "OpenEye ROCS (self)",
                "Score": float(score[0])
            })

print("\n" + "=" * 80)
print("PHASE 3: Comparative Analysis")
print("=" * 80)

if results:
    df = pd.DataFrame(results)
    
    # Separate self-scoring results from regular test results
    df_self = df[df["Molecule"].str.contains(r"\(self\)", na=False, regex=True)].copy()
    df_test = df[~df["Molecule"].str.contains(r"\(self\)", na=False, regex=True)].copy()
    
    # Display self-scoring results
    if not df_self.empty:
        print("\n🔬 Self-Scoring Results (Perfect Match Test):")
        print("   This reveals the normalization difference between scorers:")
        
        # Check for duplicate molecule names and make them unique
        if df_self["Molecule"].duplicated().any():
            df_self["Molecule"] = df_self.groupby("Molecule").cumcount().astype(str) + "_" + df_self["Molecule"]
        
        pivot_self = df_self.pivot(index="Molecule", columns="Scorer", values="Score")
        print(pivot_self.to_string())
        
        if "CDPKit (self)" in pivot_self.columns:
            cdpkit_self_mean = pivot_self["CDPKit (self)"].mean()
            print(f"\n   CDPKit Self-Score Mean: {cdpkit_self_mean:.6f} (Expected: ~2.0)")
            if abs(cdpkit_self_mean - 2.0) > 0.1:
                print(f"   ⚠️  WARNING: CDPKit self-scores deviate from expected 2.0!")
        
        if "OpenEye ROCS (self)" in pivot_self.columns:
            rocs_self_mean = pivot_self["OpenEye ROCS (self)"].mean()
            print(f"   OpenEye ROCS Self-Score Mean: {rocs_self_mean:.6f} (Expected: ~2.0)")
            if abs(rocs_self_mean - 2.0) > 0.1:
                print(f"   ⚠️  WARNING: OpenEye ROCS self-scores deviate from expected 2.0!")
        
        if "CDPKit (self)" in pivot_self.columns and "OpenEye ROCS (self)" in pivot_self.columns:
            ratio = pivot_self["OpenEye ROCS (self)"].mean() / pivot_self["CDPKit (self)"].mean()
            print(f"\n   📊 Score Ratio (OpenEye/CDPKit): {ratio:.4f}")
            if abs(ratio - 1.0) < 0.1:
                print(f"   ✅ Both scorers now return similar raw TanimotoCombo scores (0-2 range)")
            else:
                print(f"   ⚠️  Score ratio differs from 1.0 - indicates algorithmic differences")
    
    # Display test molecule results
    if not df_test.empty:
        pivot = df_test.pivot(index="Molecule", columns="Scorer", values="Score")

        print("\n📊 Test Molecule Score Comparison:")
        print(pivot.to_string())

        if "CDPKit" in pivot.columns and "OpenEye ROCS" in pivot.columns:
            pivot["Ratio"] = pivot["CDPKit"] / pivot["OpenEye ROCS"]
            pivot["Difference"] = pivot["CDPKit"] - pivot["OpenEye ROCS"]

            print("\n📈 Statistical Summary:")
            print(f"  CDPKit Mean Score:      {pivot['CDPKit'].mean():.4f}")
            print(f"  OpenEye ROCS Mean:      {pivot['OpenEye ROCS'].mean():.4f}")
            print(f"  Mean Score Ratio:       {pivot['Ratio'].mean():.4f}")
            print(f"  Mean Score Difference:  {pivot['Difference'].mean():.4f}")

            # Recommendation for threshold adjustment
            current_threshold = 0.35
            cdpkit_mean = pivot['CDPKit'].mean()
            openeye_mean = pivot['OpenEye ROCS'].mean()

            if openeye_mean > 0:
                scaling_factor = cdpkit_mean / openeye_mean
                suggested_threshold = current_threshold * scaling_factor

                print("\n" + "=" * 80)
                print("RECOMMENDATIONS")
                print("=" * 80)
                print(f"\n  Current OpenEye Threshold: {current_threshold}")
                print(f"  Scaling Factor (CDPKit/OpenEye): {scaling_factor:.4f}")
                print(f"  Suggested CDPKit Threshold: {suggested_threshold:.4f}")
                print(f"\n  🎯 To achieve similar selectivity as OpenEye with threshold 0.35,")
                print(f"     use threshold {suggested_threshold:.2f} for CDPKit scoring.")

                # Analyze score differences (both now use 0-2 range)
                if abs(scaling_factor - 1.0) < 0.2:
                    print(f"\n  ✅ Good news: Scores are similar after normalization fix!")
                    print(f"    Both CDPKit and OpenEye ROCS now use raw TanimotoCombo (0-2 range).")
                    print(f"    Algorithmic similarity appears good.")
                elif scaling_factor < 0.8:
                    print(f"\n  ⚠️  WARNING: CDPKit scores are still significantly lower than OpenEye!")
                    print(f"     Score ratio: {scaling_factor:.4f} (expected ~1.0)")
                    print(f"     This indicates differences in:")
                    print(f"       - Conformer generation quality")
                    print(f"       - Shape alignment algorithm")
                    print(f"       - Pharmacophore feature detection")
                    print(f"\n     Consider:")
                    print(f"       1. Increasing CDPKit conformer generation timeout")
                    print(f"       2. Adjusting energy window and RMSD thresholds")
                    print(f"       3. Investigating why self-scores aren't reaching ~2.0")
                else:
                    print(f"\n  ℹ️  CDPKit scores are higher than OpenEye scores.")
                    print(f"     This could indicate more lenient alignment or scoring.")
else:
    print("\n⚠️  No scoring results available for comparison")

print("\n" + "=" * 80)
print("Diagnostic Complete")
print("=" * 80)
