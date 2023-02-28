# Change Log
From v3.4.0 to v3.4.1

## Fixes

- Fixed the CLI so that it supports new QSPRPred models.

## Changes

- The `train` CLI script now uses the `'-p', '--predictor'` option to specify the QSPRPred model to use. It takes a path to the model's `_meta.json` file. More models can be specified this way.
  - This changes the original meaning of the `'-ta', '--active_targets'`, `'-ti', '--inactive_targets'` and `'-tw', '--window_targets'` options. These now serve to link the models to the particular type of target. The name of the QSPRPred model is used to determine the type of target it represents. For example, if the QSPRPred model is called `A2AR_RandomForestClassifier`, then the `'-ta', '--active_targets'` option will be used to link to the `A2AR_RandomForestClassifier` as a predictor predicting activity towards a target. 

## Removed Features

None.

## New Features

None.
