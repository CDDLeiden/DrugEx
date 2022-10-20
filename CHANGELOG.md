# Change Log
From v3.2.0 to v3.3.0

## Fixes

None.


## Changes

- Improve scaffold-based encoding. New `dummyMolsFromFragments` to create dummy molecules from set of fragments to be called as the `fragmenter` in `FragmentCorpusEncoder`. This makes the `ScaffoldSequenceCorpus`, `ScaffoldGraphCorpus`, `SmilesScaffoldDataSet` and `GraphScaffoldDataSet` classes obsolete. 
- The early stopping criterion of reinforcement learning is changed back to the ratio of desired molecules.


## New Features

- Tutorial for scaffold-based generation.
