#!/bin/bash

set -e

# Run all jupyter notebooks in the tutorial directory

# Download data/models
rm -rf data
drugex download

# Run each notebook (order matters so let's do it manually)
jupyter nbconvert --to notebook --execute Sequence-RNN.ipynb
jupyter nbconvert --to notebook --execute Graph-Transformer.ipynb
jupyter nbconvert --to notebook --execute advanced/extending_api.ipynb
jupyter nbconvert --to notebook --execute advanced/scaffold_based.ipynb
jupyter nbconvert --to notebook --execute qsar.ipynb # run this last (to test the downloaded models)