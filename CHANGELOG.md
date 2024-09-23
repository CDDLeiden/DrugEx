# Change Log
From v3.4.7 to v3.4.8

## Fixes

None.

## Changes

- `Scorer` class now supports returning multiple scores for a single prediction.
    This is useful for models that output multiple scores for a single input.

## Removed Features

None. 

## New Features

- `QSPRpredScorer` now also supports multi-task, multi-class and attached applicability
    domain predictions.
- A new tutorial on how to use multi-task scorers is now available under
    the advanced tutorials (tutorial/advanced/multitask_scorers.ipynb)
