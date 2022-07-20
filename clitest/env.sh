set -e

export PYTHONPATH=".."

# input data and base directory
export TEST_BASE="."
export TEST_DATA_PRETRAINING='ZINC_raw_small.txt'
export TEST_DATA_FINETUNING='A2AR_raw_small.txt'
export TEST_DATA_ENVIRONMENT='A2AR_raw_small_env.txt'

# prefixes for output files
export VOC_PREFIX='vocabulary'
export PRETRAINING_PREFIX='pre'
export FINETUNING_PREFIX='ft'

function cleanup() {
  rm -rf ${TEST_BASE}/data/backup_*;
  rm -rf ${TEST_BASE}/data/${FINETUNING_PREFIX}_*.txt;
  rm -rf ${TEST_BASE}/data/${FINETUNING_PREFIX}_*.vocab;
  rm -rf ${TEST_BASE}/data/${PRETRAINING_PREFIX}_*.txt;
  rm -rf ${TEST_BASE}/data/${PRETRAINING_PREFIX}_*.vocab;
  rm -rf ${TEST_BASE}/data/${VOC_PREFIX}_*.txt;
  rm -rf ${TEST_BASE}/data/*.log;
  rm -rf ${TEST_BASE}/data/*.json;
  rm -rf ${TEST_BASE}/envs;
  rm -rf ${TEST_BASE}/generators;
  rm -rf ${TEST_BASE}/logs;
}

cleanup

# default values of some common parameters
export MOL_COL='CANONICAL_SMILES'
export N_FRAGS=4
export N_COMBINATIONS=4
export FRAG_METHOD='brics'
export TRAIN_EPOCHS=2
export TRAIN_BATCH=32
export TRAIN_GPUS=0
export N_CPUS=2

###########
# DATASET #
###########

export DATASET_COMMON_ARGS="-b ${TEST_BASE} -d -mc ${MOL_COL} -sv -sif"
export DATASET_FRAGMENT_ARGS="-fm ${FRAG_METHOD} -nf ${N_COMBINATIONS} -nf ${N_FRAGS}"

###############
# ENVIRONMENT #
###############
export ENVIRON_COMMON_ARGS="-b ${TEST_BASE} -d"
python -m drugex.environ \
${ENVIRON_COMMON_ARGS} \
-i ${TEST_DATA_ENVIRONMENT} \
-l \
-s \
-m RF \
-ncpu ${N_CPUS} \
-gpu ${TRAIN_GPUS} \
-bs ${TRAIN_BATCH}

############
# TRAINING #
############
export TRAIN_COMMON_ARGS="-b ${TEST_BASE} -d -e ${TRAIN_EPOCHS} -bs ${TRAIN_BATCH} -gpu ${TRAIN_GPUS}"
export TRAIN_VOCAB_ARGS="-vfs ${PRETRAINING_PREFIX} ${FINETUNING_PREFIX}"
#export TRAIN_VOCAB_ARGS="" # uncomment to test with the default vocabularies

export RL_PREFIX="RL"
export TARGET_ID="P29274"
export ENVIRON_MODE="CLS"
export ENVIRON_ALG="RF"
export ENVIRON_THRESHOLD=6.5
export TRAIN_RL_ARGS="-ta ${TARGET_ID} -et ${ENVIRON_MODE} -ea ${ENVIRON_ALG} -at ${ENVIRON_THRESHOLD}"

