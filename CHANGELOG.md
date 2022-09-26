# Change Log
From v3.1.0 to v3.2.0

## Fixes

- fixes to SmilesExplorerNoFrag (wrong best state was saved and a TypeError while logging was eliminated, see !40)
- optimized how memory is used a little bit (!50)
- fix #55


## Changes

- generated SMILES are now not reported in the logger of SmilesExplorerNoFrag (see !40), but should still be available to the supplied training monitor

- Training QSAR models is restructured (see !41), only CLI still environ.py, actually functionality moved to environment.
As well as unittests added for this part of the code.

- Early stopping patience is now softcoded for all models and for reinforcement learning criteria was changed from ratio of desired molecules to the mean average score (see !46)


## New Features

- add option to remove molecules with tokens not occuring in voc (in dataset.py), see !39.

- add grid search for DNN QSAR model (see !41)
- add bayes optimization for DNN QSAR model (see !42)

- add option to use different environment algorithms during RL
- add option to use selectivity window predictor for RL

- add option to specify `chunk_size` in the `dataset.py` script to control how data is supplied to parallel processes (bigger chunk size -> more memory used, but more efficient use of multiple CPUs, see !50)
