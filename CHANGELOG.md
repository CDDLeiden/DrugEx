# Change Log
From v3.4.0 to v3.4.1

## Fixes

- Content of output files during model training and molecule generation (broken due to refactoring in `v3.4.0`):
  - During fine-tuning, the training (`train_loss`) and the validation (`valid_loss`) loss, the rations of valid (`valid_ratio`) and accurate (`accurate_ratio`, only for transformers) molecules are saved in `_fit.tsv`
  - During RL, the rations of valid (`valid_ratio`), accurate (`accurate_ratio`, only for transformers), unique (`unique_ratio`) and desired (`desired_ratio`) molecules and the average arithmetic (`avg_amean`) and geometric (`avg_gmean`) of the modified scores are saved in `_fit.tsv`
- In `DrugExEnvironment.getScores()` set all modified scores to 0 for invalid molecules (fixes bug resulting from refactoring in `v3.4.0`)
- Fixed the CLI so that it supports new QSPRPred models.

## Changes

- The `train` CLI script now uses the `'-p', '--predictor'` option to specify the QSPRPred model to use. It takes a path to the model's `_meta.json` file. More models can be specified this way.
  - This changes the original meaning of the `'-ta', '--active_targets'`, `'-ti', '--inactive_targets'` and `'-tw', '--window_targets'` options. These now serve to link the models to the particular type of target. The name of the QSPRPred model is used to determine the type of target it represents. For example, if the QSPRPred model is called `A2AR_RandomForestClassifier`, then the `'-ta', '--active_targets'` option will be used to link to the `A2AR_RandomForestClassifier` as a predictor predicting activity towards a target. 
- Standard crowding distance is now the default ranking method for the `train` script (equiv. to `--scheme PRCD`, previously was `--scheme PRTD`).

## Removed Features

None.

## New Features

None.
