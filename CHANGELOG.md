# Change Log
From v3.4.3 to v3.4.4

## Fixes

- Fixed a bug that may have caused the standardizer to return molecules failing in standardization in their original form instead of removing them (14fd58dc758cb882c2a24e4a481a9064318927f1).

## Changes

- In `SequenceExplorer`, the nSamples parameter is now the number of molecules sampled for the policy update and an
  additional 10% is used for performance evaluation. Previously the number sampled was 10 times the batch size + nSamples
  for evaluation.
- `train_loader` is now an optional argument for the `SequenceExplorer`. Both `train_loader` and `test_loader` are not 
  used by `SequenceExplorer`, this is unchanged, but has now been clarified in the documentation.

## Removed Features

None.

## New Features

None.
