# Change Log
From v3.4.4 to v3.4.5

## Fixes

- Fixed a bug in calculation of the Pareto fronts (fronts are now calculated for maximization of objectives instead of objective minimization).
- Patch a bug that that caused a crash when an invalid smiles was encountered in the fragment generation step. This
  bug was introduced in v3.4.4, now invalid smiles are skipped and a warning is printed to the log.

## Changes

- Installation of pip package with pyproject.toml instead of setup.cfg.
- Methods `cpu_non_dominated_sort` and `gpu_non_dominated_sort` have been replace by `get_Pareto_fronts`.
- Improve calculation of crowding distance.
- The rewards module is refactored and the `RankingStrategy` class was replace by `ParetoRankingScheme` class. 
    - The final reward calcuation for `ParetoRankingScheme`-based methods is now directly the scaled rank of the molecules.
    - The `ParetoTanimotoDistance` now has a attribute `distance_metric` which can be "min", "mean" or "mutual" instead of attribute `ranking`.
- DrugEx is now compatible with the latest version of qsprpred v2.0.1, previous versions of qsprpred are no longer supported.

## Removed Features

None. 

## New Features

- When installing package with pip, the commit hash and date of the installation is saved into `drugex._version`. This information is also used as a basis of a new dynamic versioning scheme for the package. The version number is generated automatically upon installation of the package and saved to `drugex.__version__`. 
