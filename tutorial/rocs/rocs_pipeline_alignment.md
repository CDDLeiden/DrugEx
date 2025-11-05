# ROCS De Novo Pipeline Comparison

## Quick Takeaways
- Original scripts ran long, low-epsilon RL with unclipped ROCS scores and a 0.35 ROCS threshold (`tutorial/rocs/original_drugex_rocs/DrugEx_script.py:27`, `tutorial/rocs/original_drugex_rocs/DrugEx_script.py:52`).
- The DrugEx_ROCS_RNN_Showcase notebook shortens RL, raises ε, clips ROCS scores at 0.9, and sometimes lowers the ROCS threshold to 0.20 (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:688`, `tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:901`, `tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:822`).
- Both pipelines use `model3-4_v1.sq`, but the notebook evaluates both the `.sq` and an SDF query, taking the maximum score (`tutorial/rocs/original_drugex_rocs/DrugEx_script.py:45`, `tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:707`).

## Side-by-Side Comparison
| Aspect | Original Pipeline | Notebook Default | Difference & Impact |
| --- | --- | --- | --- |
| RL epochs & patience | 1 000 epochs, patience 1 000 (`tutorial/rocs/original_drugex_rocs/DrugEx_script.py:27`) | 30 epochs, no explicit patience (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:688`) | Less training and no patience makes convergence unlikely; major impact. |
| Exploration rate (ε) | 0.01 (`tutorial/rocs/original_drugex_rocs/DrugEx_script.py:27`) | 0.1 (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:688`) | Higher ε keeps exploration high; slows exploitation of good leads. |
| Sample budget | 10 000 per epoch (`tutorial/rocs/original_drugex_rocs/DrugEx_script.py:27`) | 1 000 per epoch (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:688`) | Fewer samples reduce learning signal; strong effect. |
| Batch size | 128 (`tutorial/rocs/original_drugex_rocs/DrugEx_script.py:27`) | 256 (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:688`) | Larger batch decreases update frequency; moderate effect. |
| ROCS score modifier | None (raw TanimotoCombo) (`tutorial/rocs/original_drugex_rocs/DrugEx_script.py:52`) | `SmoothClippedScore(upper_x=0.9, lower_x=0.4)` (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:901`) | Clipping compresses 0–2 range into ~0–1; flattens gradients; major impact. |
| SA score modifier | `SmoothClippedScore(lower_x=5, upper_x=3)` (`tutorial/rocs/original_drugex_rocs/DrugEx_script.py:52`) | `SmoothClippedScore(lower_x=5.0, upper_x=3.0)` (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:909`) | Same behaviour. |
| Thresholds | [0.35 (ROCS), 0.1 (SA)] (`tutorial/rocs/original_drugex_rocs/DrugEx_script.py:59`) | [0.35] aggregates, [0.20] supermol; SA 0.1 appended (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:802`, `tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:822`, `tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:926`) | Supermol runs accept scores down to 0.20; dilutes reward; moderate impact. |
| ROCS query files | Single `model3-4_v1.sq` (`tutorial/rocs/original_drugex_rocs/DrugEx_script.py:45`) | `model3-4_v1.sq` plus `CCR2_reference_ligands.sdf` (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:707`) | Taking max of two references broadens objectives; moderate impact. |
| Conformer generator | Omega defaults; skip >45 heavy atoms; ≤4 stereoisomers (`tutorial/rocs/original_drugex_rocs/Model_scorer.py:21`, `tutorial/rocs/original_drugex_rocs/Model_scorer.py:46`) | Omega/RDKit/CDPKit with max_conformers=40, heavy atom limit 35, max_centers=4 (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:715`, `drugex/training/scorers/conformer_generators.py:67`) | Stricter filter removes larger ligands; 40 conformers can inflate scores; moderate impact. |
| Scorer implementation | Custom OpenEye `ModelScorer` (`tutorial/rocs/original_drugex_rocs/Model_scorer.py`) | Modular CLIROCS, RDKit, CDPKit scorers (`drugex/training/scorers/cli_rocs.py`, `drugex/training/scorers/rdkit_rocs.py`, `drugex/training/scorers/cdpkit_rocs.py`) | Feature-rich but behaviour differs; needs parameter alignment. |

## Detailed Findings

### Reinforcement Learning Configuration
- **Original references** (`tutorial/rocs/original_drugex_rocs/DrugEx_script.py:27–32`):
  - `N_SAMPLES = 10000`
  - `EPSILON = 0.01`
  - `BATCH_SIZE = 128`
  - `PATIENCE = 1000`
  - `N_EPOCHS = 1000`
- **Notebook defaults** (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:688–691`):
  - `epochs = 30`
  - `epsilon = 0.1`
  - `n_samples = 1000`
  - `batch_size = 256`
  - `reload_interval = 25`
- **Explanation**: The new pipeline performs 33× fewer epochs, samples 10× fewer molecules per epoch, and explores 10× more aggressively. Combined with the larger batch size (fewer updates), the agent under-fits and cannot exploit promising regions, which directly degrades generation quality.

### Scorers and Score Modifiers
- **Original** (`tutorial/rocs/original_drugex_rocs/DrugEx_script.py:45–64`):
  - Single `ModelScorer` using `qfnames=['model3-4_v1.sq']`, `score='TanimotoCombo'`.
  - `Property("SA")` with `SmoothClippedScore(lower_x=5, upper_x=3)`.
  - Thresholds `[0.35, 0.1]`.
  - No modifier applied to ROCS scores (raw TanimotoCombo fed into environment).
- **Notebook** (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:894–914`):
  - Automatically attaches `SmoothClippedScore(upper_x=0.9, lower_x=0.4)` to shape scorers that lack modifiers.
  - Always appends `Property("SA")` with `SmoothClippedScore(lower_x=5.0, upper_x=3.0)` if missing.
  - If thresholds length < number of scorer keys, appends `SA_THRESHOLD = 0.1`.
- **Explanation**: Clipping the ROCS score between approximately 0 and 1 with midpoint 0.65 erases the gradient signal from high-quality alignments (TanimotoCombo can reach 2.0). The SA handling remains consistent, but the automatic clipping is a significant behavioural drift.

### Threshold Policies
- **Original** (`tutorial/rocs/original_drugex_rocs/DrugEx_script.py:59–64` and `tutorial/rocs/original_drugex_rocs/generate_scoring.py:18–28`): `thresholds = [0.35, 0.1]`.
- **Notebook by backend** (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:802–842`, `tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:926–932`):
  - OpenEye, RDKit aggregate, CDPKit aggregate set `[0.35]` → after SA append the effective list is `[0.35, 0.1]`.
  - RDKit supermol, CDPKit supermol set `[0.20]` → effective `[0.20, 0.1]`.
- **Explanation**: Allowing ROCS scores down to 0.20 lowers the quality bar and alters Pareto dominance. For the aggregate backends the thresholds match, but for supermolecule runs this change encourages retention of weak overlays.

### Conformer and Isomer Generation
- **Original pipeline** (`tutorial/rocs/original_drugex_rocs/Model_scorer.py`):
  - Enumerates stereoisomers, skipping molecules with >4 enumerated isomers, >15 rotatable bonds, or >45 heavy atoms (`tutorial/rocs/original_drugex_rocs/Model_scorer.py:21–44`).
  - Uses OpenEye Omega via `OESimpleAppOptions` without explicitly setting `max_confs`; relies on defaults (`tutorial/rocs/original_drugex_rocs/Model_scorer.py:46–83`).
  - ROCS processing reads `.sq` queries and returns max TanimotoCombo per molecule (`tutorial/rocs/original_drugex_rocs/Model_scorer.py:95–173`).
- **Notebook conformer generators**:
  - Omega: `OmegaConformerGenerator(max_conformers=40, show_progress=True)` (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:715`), with internal heavy atom limit 35 and rotatable bond limit 15 (`drugex/training/scorers/conformer_generators.py:67`).
  - RDKit: `RDKitConformerGenerator(max_conformers=40)` (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:729`, `tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:742`).
  - CDPKit: `CDPKitConformerGenerator(max_conformers=40, max_centers=4, timeout=3600, min_rmsd=0.5, energy_window=20.0)` (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:880–902`).
- **Explanation**: The notebook’s 35 heavy atom cutoff excludes molecules the original pipeline would process. Generating 40 conformers per molecule increases maximum possible scores, widening variance and shifting reward distributions. The original’s implicit Omega defaults likely produced fewer conformers and allowed heavier ligands.

### Scorer Implementations
- **Original**: bespoke `ModelScorer` (OpenEye) storing results to CSV, handling isomer grouping manually (`tutorial/rocs/original_drugex_rocs/Model_scorer.py`).
- **Notebook**: modular scorers
  - `CLIROCSScorer` (OpenEye CLI) with support for multiple queries, `-rankby TanimotoCombo`, `-opt true`, `-optchem true` (`drugex/training/scorers/cli_rocs.py:182–214`).
  - `RDKitROCSScorer` combining shape+color; supports aggregate versus supermolecule modes (`drugex/training/scorers/rdkit_rocs.py`).
  - `CDPKitROCSScorer` aligning with CDPKit shapescreen defaults (`drugex/training/scorers/cdpkit_rocs.py`).
- **Explanation**: The new setup is more flexible but that latitude means parameter mismatches (like clipping and thresholds) easily slip in. For strict reproduction, matching the original OpenEye-only workflow is essential.

### ROCS Query Assets
- **Shared**: `tutorial/rocs/rocs_rl_ccr/rdkit_cdpkit/model3-4_v1.sq` is present and used in both (`tutorial/rocs/original_drugex_rocs/DrugEx_script.py:45`, `tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:164`).
- **Notebook additions**: `CCR2_reference_ligands.sdf` and other references located under `RESOURCES_DIR` (`tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:162–166`).
- **Explanation**: The new OpenEye backend combines `.sq` and SDF queries, selecting the best result per molecule. To match original behaviour, restrict to the `.sq` file.

## What Changed & Why Results Diverged
- **Short RL schedule + high ε**: The notebook trains for only 30 epochs with more exploration, so the agent never stabilizes near high-scoring regions. This is the single largest deviation.
- **ROCS score clipping**: Saturating TanimotoCombo at 0.9 removes reward differentiation among good poses, starving the policy of gradient information.
- **Threshold shift to 0.20** (supermol backends): Allows weak overlays into the Pareto front, reducing pressure to achieve high ROCS scores.
- **Conformer policy**: Capping heavy atoms at 35 excludes molecules the original workflow scored; meanwhile up to 40 conformers increases max-score variance.
- **Dual-query scoring**: Taking the best of an SDF and .sq query changes the optimisation landscape versus the original single-query ROCS scoring.

## Alignment Checklist
- **Match RL hyperparameters**: Set `epochs=1000`, `epsilon=0.01`, `n_samples=10000`, `batch_size=128`, `patience=1000` (if available).
- **Remove shape clipping**: Skip the automatic `SmoothClippedScore` on ROCS scorers; keep only the SA modifier.
- **Standardize thresholds**: Use `thresholds=[0.35]` for every backend so the helper appends `0.1` for SA.
- **Relax conformer filters**: Allow up to 45 heavy atoms and consider lowering `max_conformers` to the original Omega-effective count.
- **Single-query scoring**: Restrict OpenEye backend to `model3-4_v1.sq` if you want legacy behaviour.

## Shared Components
- Both pipelines rely on `tutorial/rocs/rocs_rl_ccr/rdkit_cdpkit/model3-4_v1.sq` (`tutorial/rocs/original_drugex_rocs/DrugEx_script.py:45`, `tutorial/rocs/DrugEx_ROCS_RNN_Showcase.ipynb:164`).
- SA scoring uses the same `SmoothClippedScore(lower_x=5.0, upper_x=3.0)` modifier.

## To-Do for Full Parity (Optional edits)
1. Update `RL_DEFAULTS` in the notebook to `epochs=1000`, `epsilon=0.01`, `n_samples=10000`, `batch_size=128`, `reload_interval` large enough to mimic patience.
2. Disable automatic ROCS clipping in `build_environment_and_modifier`; leave only SA clipping.
3. Set every backend’s `thresholds` to `[0.35]`.
4. Increase Omega/RDKit/CDPKit heavy atom limits to 45 and reduce `max_conformers` to align with Omega defaults (e.g., 10–20).
5. Configure OpenEye backend `query_files={"Model3-4_SQ": str(ROCS_SQ_QUERY)}`.
6. Confirm thresholds expand to `[0.35, 0.1]` and that SA modifier remains `SmoothClippedScore(lower_x=5.0, upper_x=3.0)`.

Applying these adjustments will reproduce the original reward surface and training dynamics, making comparison of old versus new results meaningful.
