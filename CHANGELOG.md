# Change Log
From v3.4.2 to v3.4.3

## Fixes

- The `dataset` CLI script now captures test, train and unique sets in the right order for the fragment-based methods.
- Examples in the CLI tutorial were updated and fixed.
- The Linux command line script was fixed so that all arguments are passed correctly.
- The `train` and `generate` CLI scripts now have empty predictor (`-p, --predictor`) by default. This makes error messages less confusing.
- Fixes bug in desirability calculation during generation

## Changes

- In `SequenceExplorer`, the nSamples parameter is now the number of molecules sampled for the policy update and an
  additional 10% is used for performance evaluation. Previously the number sampled was 10 times the batch size + nSamples
  for evaluation.
- `train_loader` is now an optional argument for the `SequenceExplorer`. Both `train_loader` and `test_loader` are not 
  used by `SequenceExplorer`, this is unchanged, but has now been clarified in the documentation.

## Removed Features

None.

## New Features

- New fragmenter `FragmenterWithSelectedFragment` which produces only fragments-molecule pair in which the input fragments contain the specific fragment given by the user
