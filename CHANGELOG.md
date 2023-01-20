# Change Log
From v3.2.0 to v3.3.0

## Fixes

None.


## Changes

- Improve scaffold-based encoding. New `dummyMolsFromFragments` to create dummy molecules from set of fragments to be called as the `fragmenter` in `FragmentCorpusEncoder`. This makes the `ScaffoldSequenceCorpus`, `ScaffoldGraphCorpus`, `SmilesScaffoldDataSet` and `GraphScaffoldDataSet` classes obsolete. 
- The early stopping criterion of reinforcement learning is changed back to the ratio of desired molecules.
- Renamed `GaphModel.sampleFromSmiles` to `GraphModel.sample_smiles`,
  - argument `min_samples` was renamed to `num_samples`,
  - exactly `num_samples` are returned,
  - arguments `drop_duplicates`, `drop_invalid` were added,
  - argument `keep_frags` was added.
- The `sample_smiles` method was added to the SequenceTranformer `GTP2Model` and to the `RNN` classes.
- Changed the `GTP2Model` adaptive learning rate settings to resolve pretraining issues
- Progress bars were added for models' fitting (pretraining, fine-tuning and reinforcement learning).
- Tokens `_` and `.` always present in `VocSmiles` have been removed.
- RNN models deposited on Zenodo and pretrained on ChEMBL31 and Papyrus 05.5 were updated while the RNN model pretrained on ChEMBL27 did not need to.
- Moved encoding of tokens for SMILES-based models to the parallel preprocessing steps to improve performance
- All testing code that is not unit tests was moved to `testing`
- Remove QSAR modelling from DrugEx, now in a seperate repository QSPRpred
- Revised SimilarityRanking, now uses the minimum (default) or average Tanimoto distance to rank the solutions in the same front.
- QSPRpred is an optional dependency (only required for the CLI)
- `Fragmenter` can now be instructed not to skip molecules resulting in only one fragment with the `allow_single` argument.


## New Features

- Tutorial for scaffold-based generation.
- Added tests to `testing` that allow to check consistency of models between versions.
