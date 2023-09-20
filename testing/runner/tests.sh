#!/bin/bash

set -e

CONDA_ROOT=/opt/conda/bin
ACTIVATE_CMD="source ${CONDA_ROOT}/activate"
ENV_NAME="drugex"
RUN_CMD="${ACTIVATE_CMD} && conda activate ${ENV_NAME}"
WD=`pwd`

# setting up environments
echo "Creating environment: ${ENV_NAME}"
bash -c "${ACTIVATE_CMD} && conda create -n ${ENV_NAME} python=${PYTHON_VERSION}"

echo "Installing drugex package..."
bash -c "${RUN_CMD} && pip install git+${DRUGEX_REPO}@${DRUGEX_REVISION}"

echo "Installing qsprpred package..."
bash -c "${RUN_CMD} && pip install git+${QSPRPRED_REPO}@${QSPRPRED_REVISION}"

echo "Checking for CUDA..."
bash -c "${RUN_CMD} && python -c 'import torch; print(torch.cuda.is_available())'"
echo "Checking for drugex version..."
bash -c "${RUN_CMD} && python -c 'import drugex; print(drugex.__version__)'"
echo "Checking for qsprpred version..."
bash -c "${RUN_CMD} && python -c 'import qsprpred; print(qsprpred.__version__)'"

# running tests
echo "Running unit tests..."
bash -c "${RUN_CMD} && python -m unittest discover drugex"

echo "Running CLI tests..."
git clone ${DRUGEX_REPO}
cd DrugEx
git checkout ${DRUGEX_REVISION}
cd testing/clitest
bash -c "${RUN_CMD} && ./test.sh"

echo "Installing tutorial dependencies..."
bash -c "${RUN_CMD} && pip install papyrus_structure_pipeline git+https://github.com/martin-sicho/papyrus-scaffold-visualizer.git@main mols2grid jupyterlab"
bash -c "${RUN_CMD} && pip install git+${QSPRPRED_REPO}@${QSPRPRED_REVISION}" # ensure version

echo "Running tutorials..."
cd "${WD}/DrugEx/tutorial"
bash -c "${RUN_CMD} && ./run_all.sh"

echo "All tests finished successfully. Exiting..."
