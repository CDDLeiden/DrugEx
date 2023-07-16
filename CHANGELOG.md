# Change Log
From v3.4.4 to v3.4.5

## Fixes

- Fixed a bug in calculation of the Pareto fronts (fronts are now calculated for maximization of objectives instead of objective minimization).

## Changes

- Methods `cpu_non_dominated_sort` and `gpu_non_dominated_sort` have been replace by `get_Pareto_fronts`.
- Improve calculation of crowding distance.

## Removed Features

None. 

## New Features

None.
