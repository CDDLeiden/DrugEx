# Change Log

Current: v3.2.0.dev0

Previous: v3.1.0

## Fixes

- fixes to `SmilesExplorerNoFrag` (wrong best state was saved and a TypeError while logging was eliminated, see !40)

## Changes

- generated SMILES are now not reported in the logger of `SmilesExplorerNoFrag` (see !40), but should still be available to the supplied training monitor

## New Features

- add option to remove molecules with tokens not occuring in voc (in dataset.py), see !39.
