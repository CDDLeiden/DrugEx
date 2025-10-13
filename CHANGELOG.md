# Change Log
From v3.4.8 to v3.4.9

## Fixes

- Fix broken link to quick start tutorial in README

## Changes

- For the `SequenceExplorer` the current epoch and the maximum number of rl epochs are 
  now logged to the `{model_name}_fit.log` file.
- Added `weights_only=True` in `torch.load` to the loading in `Model.loadStatesFromFile` so
  the warning about arbitrary code execution from PyTorch is no longer raised.

## Removed Features

None. 

## New Features

- Add argument 'loss_tolerance' to `Generator.fit`. Setting a loss tolerance allows 
  a small increase in the training loss before stopping training early if the
  `valid_fraction` is increased.
