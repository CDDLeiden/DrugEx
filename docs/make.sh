#!/usr/bin/env bash

# stop if anything goes wrong
set -e

rm -rf _build

export PYTHONPATH=`pwd`/../

# configure
sphinx-apidoc -o ./api/ ../drugex/

# make
make html
