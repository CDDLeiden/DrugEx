# Command Line Interface Tests

In this folder, the CLI test suite can be run. Not all combinations of parameters are currently tested, but the most important ones are. You can run tests for all models from this folder as follows:

```bash
# if you do not have the package installed and only the dependencies
PYTHONPATH=../.. ./test.sh

# if you have the package installed
./test.sh
```