# Change Log
From v3.4.5 to v3.4.6

## Fixes

None.

## Changes

- For the generator CLI, environment variables are now read from the generator meta file automatically. Now unused arguments are removed from the CLI.
- Compatibility updates to make the package work with the latest QSPRpred scorers in version 3.0.0 and higher. Older scorers will still work if an older version is installed alongside DrugEx. Only the unit tests will fail since the models used there assume QSPPRpred v3.0.0 or later.

## Removed Features

None. 

## New Features

None.
