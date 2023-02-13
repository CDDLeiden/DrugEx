# Change Log
From v3.3.0 to v3.4.0

## Fixes

None.


## Changes

Major refactoring of `drugex.training`

- Moving generators from `drugex.training.models` to `drugex.training.generators`, and harmonizing and renaming them
  - `RNN` -> `SequenceRNN`
  - `GPT2Model` -> `SequenceTransformer`
  - `GraphModel` -> `GraphTransformer`

- Moving explorers from `drugex.training.models` to `drugex.training.explorers`, harmonizing and renaming them
  - `SmilesExplorerNoFrag` -> `SequenceExplorer`
  - `SmilesExplorer` -> `FragSequenceExplorer`
  - `GraphExplorer` -> `FragGraphExplorer`

- Removal of all obsolete modules related to the two discontinued fragment-based LSTM models from [DrugEx v3](https://doi.org/10.26434/chemrxiv-2021-px6kz).

- The generators' `sample_smiles()` has been replaced by a `generate()` function

- Clafification of the terms qualifying the generated molecules to have the following unique and constant definitions (replacing ambigous `VALID` and `DESIRE` terms)
  - `Valid` : molecule can be parsed with rdkit
  - `Accurate` : molecule contains given input fragments
  - `Desired` : molecule fulfils all given objectives 


- Revise implementation of Tanimoto distance-based Pareto ranking scheme(`SimilarityRanking`) to correspond to the method described in [DrugEx v2](https://doi.org/10.1186/s13321-021-00561-9). Add option to use minimum Tanimoto distance between molecules in a front instead the mean distance.

- Remove all references to NN-based RAscore (already discontinued)

Refactoring of CLI

- Refactoring `dataset.py` and `train.py` to object based
- Writting a single `.txt.vocab` file per dataset preprocessing instead of separate (duplicate) files for each subset in `dataset.py`

## Removed

- `--save_voc` argument in `dataset.py` as redundant
- `--pretrained_model` argment in `train.py` (merged with `--agent_path`)


## New Features

- GRU-based RNN added to the CLI 
- added another possible implementation of similarity ranking (`MutualSimilaritySortRanking`), this is based on the code in the original repository of [DrugEx](https://github.com/XuhanLiu/DrugEx/blob/cd384f4a8ed4982776e92293f77afd4ea78644f9/utils/nsgaii.py#L92)
