# Change Log
From v3.4.1 to v3.4.2

## Fixes

- The `QSPRPredScorer` now functions properly when presented with rdkit molecules instead of SMILES strings. It also does not modify the input list anymore.

## Changes

None.

## Removed Features

None.

## New Features

- New fragmenter `FragmenterWithSelectedFragment` which produces only fragments-molecule pair in which the input fragments contain the specific fragment given by the user
