#!/bin/bash

# This script is used to initialize the environment in a Google Colab notebook and pull tutorial data automatically.
# Just run it with `!bash colab.sh` in a Colab notebook.

# Download and install dependencies
wget https://raw.githubusercontent.com/CDDLeiden/DrugEx/master/tutorial/requirements.txt
pip install -r requirements.txt

# Download data
drugex download