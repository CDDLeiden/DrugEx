# Change Log
From v3.4.8 to v3.4.9

## Fixes

- Fix broken link to quick start tutorial in README
- Fix a crash of `SequenceRNN` on CPU-only machines: the `-1` "no GPU" sentinel was
  formatted into the device string `cuda:-1`.
- Fix the ROCS tutorial: `config.py` resolved the pretrained model one directory short,
  `prepare_models.py` depended on the working directory, and fine-tuning ran after the
  setup cell that needs it, so the notebook could not run top to bottom.
- Scorer tests in `drugex/training/scorers/tests/` were never collected by
  `python -m unittest discover`, because the directory had no `__init__.py`. They now run.

## Changes

- For the `SequenceExplorer` the current epoch and the maximum number of rl epochs are 
  now logged to the `{model_name}_fit.log` file.
- Added `weights_only=True` in `torch.load` to the loading in `Model.loadStatesFromFile` so
  the warning about arbitrary code execution from PyTorch is no longer raised.
- The shape matching scorers now choose what the alignment optimizes, as `shape`, `combo`
  or `color`: `score_type` for `RDKitROCSScorer` (which previously only chose the reported
  score), `opt_mode` for `CDPKitROCSScorer` and `optimization_mode` for `OpenEyeROCSScorer`.
- `RDKitROCSScorer` scores no longer depend on the input pose of the probe
  (see rdkit/rdkit#8513). Scores and calibrated thresholds from earlier versions may shift
  slightly.
- The RDKit, CDPKit and OMEGA conformer generators now protonate (Dimorphite-DL) and
  canonicalize tautomers by default (`protonate`, `canonicalize_tautomer`), and bound
  conformer generation with a `timeout` of 20 s (previously 3600 s for CDPKit). Molecules
  that are not already protonated may therefore score differently.
- The scorer tests are collected in `drugex/training/scorers/tests.py`.

## Removed Features

None. 

## New Features

- Add argument 'loss_tolerance' to `Generator.fit`. Setting a loss tolerance allows 
  a small increase in the training loss before stopping training early if the
  `valid_fraction` is increased.
- Shape matching as an objective function was added (see [tutorial](./tutorial/advanced/rocs))
- Added a Docker container to easily setup DrugEx and QSPRpred enabled environments. See [Dockerfile](./docker/) for details.
- Tversky score variants for all three ROCS scorers (`score_variant="tversky_ref"` or
  `"tversky_fit"`).
- OpenEye ImplicitMillsDean color types for the RDKit and CDPKit scorers
  (`color_ff="oe_mills_dean"`). `RDKitROCSScorer(inject_color=True)` adds the hydrophobe
  color type.
- `OpenEyeROCSScorer` gains a per-call `timeout` and MPI parallelism (`mpi_np`).
