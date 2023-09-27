set -e

# Terminal formatting
export line=$(printf -vl "%${COLUMNS:-`tput cols 2>&-||echo 80`}s\n" && echo ${l// /-})

# input data and base directory
export TEST_BASE="."
export TEST_DATA_PRETRAINING='ZINC_raw_small.tsv'
export TEST_DATA_FINETUNING='A2AR_raw_small.tsv'
export TEST_DATA_SCAFFOLD='pyrazines.tsv'

# prefixes for output files
export VOC_PREFIX='vocabulary'
export PRETRAINING_PREFIX='pre'
export GRU_PREFIX='gru'
export FINETUNING_PREFIX='ft'
export SCAFFOLD_PREFIX='scaffold'

function cleanup() {
  rm -rf ${TEST_BASE}/data/backup_*;
  rm -rf ${TEST_BASE}/data/${FINETUNING_PREFIX}_*.tsv;
  rm -rf ${TEST_BASE}/data/${FINETUNING_PREFIX}_*.txt;
  rm -rf ${TEST_BASE}/data/${FINETUNING_PREFIX}_*.vocab;
  rm -rf ${TEST_BASE}/data/${PRETRAINING_PREFIX}_*.tsv;
  rm -rf ${TEST_BASE}/data/${PRETRAINING_PREFIX}_*.txt;
  rm -rf ${TEST_BASE}/data/${PRETRAINING_PREFIX}_*.vocab;
  rm -rf ${TEST_BASE}/data/${VOC_PREFIX}_*.txt;
  rm -rf ${TEST_BASE}/data/${SCAFFOLD_PREFIX}_*;
  rm -rf ${TEST_BASE}/data/*.log;
  rm -rf ${TEST_BASE}/data/*.json;
  rm -rf ${TEST_BASE}/generators;
  rm -rf ${TEST_BASE}/logs;
  rm -rf ${TEST_BASE}/new_molecules
}

# default values of some common parameters
export MOL_COL='CANONICAL_SMILES'
export N_FRAGS=4
export N_COMBINATIONS=4
export FRAG_METHOD='brics'
export TRAIN_EPOCHS=2
export TRAIN_BATCH=32
export TRAIN_GPUS=0
export N_CPUS=2
export OPTIMIZATION='bayes'
export N_TRIALS=2

###########
# DATASET #
###########
export DATASET_COMMON_ARGS="-b ${TEST_BASE} -d -mc ${MOL_COL} -sif"
export DATASET_FRAGMENT_ARGS="-fm ${FRAG_METHOD} -nc ${N_COMBINATIONS} -nf ${N_FRAGS}"

############
# TRAINING #
############
export TRAIN_COMMON_ARGS="-b ${TEST_BASE} -d -e ${TRAIN_EPOCHS} -bs ${TRAIN_BATCH} -gpu ${TRAIN_GPUS}"
export TRAIN_VOCAB_ARGS="-vfs ${PRETRAINING_PREFIX} ${FINETUNING_PREFIX}"
#export TRAIN_VOCAB_ARGS="" # uncomment to test with the default vocabularies

export ENV_MODEL_NAME="A2AR_RandomForestClassifier"
export ENV_MODEL_PATH="qsar/${ENV_MODEL_NAME}/${ENV_MODEL_NAME}_meta.json"
export TRAIN_RL_ARGS="-p ${ENV_MODEL_PATH} -ta ${ENV_MODEL_NAME}"

############
# GENERATE #
############
export DESIGN_COMMON_ARGS="-b ${TEST_BASE} -d -gpu ${TRAIN_GPUS} -n 10 -bs ${TRAIN_BATCH}"