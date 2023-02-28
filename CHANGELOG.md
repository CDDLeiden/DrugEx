# Change Log
From v3.4.0 to v3.4.1

## Fixes

- Content of output files during model training and molecule generation (broken due to refactoring in `v3.4.0`):
  - During fine-tuning, the training (`train_loss`) and the validation (`valid_loss`) loss, the rations of valid (`valid_ratio`) and accurate (`accurate_ratio`, only for transformers) molecules are saved in `_fit.tsv`
  - During RL, the rations of valid (`valid_ratio`), accurate (`accurate_ratio`, only for transformers), unique (`unique_ratio`) and desired (`desired_ratio`) molecules and the average arthitmetic (`avg_amean`) and geometric (`avg_gmean`) of the modified scores are saved in `_fit.tsv`
- In `DrugExEnvironment.getScores()` set all modified §scores to 0 for invalid molecules (fixes bug induced in (broken due to refactoring in `v3.4.0`)

<!-- ## Changes

## Removed

## New Features -->