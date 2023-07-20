# Change Log
From v3.4.4 to v3.4.5

## Fixes

- Fixed a bug in calculation of the Pareto fronts (fronts are now calculated for maximization of objectives instead of objective minimization).

## Changes

- Methods `cpu_non_dominated_sort` and `gpu_non_dominated_sort` have been replace by `get_Pareto_fronts`.
- Improve calculation of crowding distance.
- The rewards module is refactored and the `RankingStrategy` class was replace by `ParetoRankingScheme` class. 
    - The final reward calcuation for `ParetoRankingScheme`-based methods is now directly the scaled rank of the molecules.
    - The `ParetoTanimotoDistance` now has a attribute `distance_metric` which can be "min", "mean" or "mutual" instead of attribute `ranking`.

## Removed Features

None. 

## New Features

None.
