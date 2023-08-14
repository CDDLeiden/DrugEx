#!/bin/bash

set -e

# Run all jupyter notebooks in the tutorial directory

# Install requirements
# pip install mols2grid
# pip install git+https://github.com/martin-sicho/papyrus-scaffold-visualizer.git@dev#egg=papyrus-scaffold-visualizer

# Download data/models
# python -m drugex.download

# Run each notebook (order matters so let's do it manually)
jupyter nbconvert --to notebook --execute qsar.ipynb
jupyter nbconvert --to notebook --execute Sequence-RNN.ipynb
jupyter nbconvert --to notebook --execute Graph-Transformer.ipynb
jupyter nbconvert --to notebook --execute advanced/extending_api.ipynb
jupyter nbconvert --to notebook --execute advanced/scaffold_based.ipynb
