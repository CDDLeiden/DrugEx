#!/bin/bash

set -e

CONDA_ROOT=/opt/conda/bin
ACTIVATE_CMD="source ${CONDA_ROOT}/activate"
ENV_NAME="drugex"
RUN_CMD="${ACTIVATE_CMD} && conda activate ${ENV_NAME}"
WD=`pwd`

# setting up environments
echo "Setting up channels..."
conda config --remove-key channels
conda config --add channels conda-forge
conda config --show channels
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
echo "Creating environment: ${ENV_NAME}"
bash -c "${ACTIVATE_CMD} && conda create -n ${ENV_NAME} python=${PYTHON_VERSION} gcc"

echo "Installing drugex package and jupyterlab..."
bash -c "${RUN_CMD} && pip install git+${DRUGEX_REPO}@${DRUGEX_REVISION} jupyterlab git+${QSPRPRED_REPO}@${QSPRPRED_REVISION}"
