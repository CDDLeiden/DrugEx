#!/bin/bash

## Test some common command line options automatically.

set -e

export PYTHONPATH=".."

# input data and base directory
TEST_BASE="."
TEST_DATA_PRETRAINING='ZINC_raw_small.txt'
TEST_DATA_FINETUNING='A2AR_raw_small.txt'
TEST_DATA_ENVIRONMENT='A2AR_raw_small_env.txt'

# prefixes for output files
VOC_PREFIX='vocabulary'
PRETRAINING_PREFIX='pre'
FINETUNING_PREFIX='ft'

# default values of some common parameters
MOL_COL='CANONICAL_SMILES'
N_FRAGS=4
N_COMBINATIONS=4
FRAG_METHOD='brics'
TRAIN_EPOCHS=2
TRAIN_BATCH=32
TRAIN_GPUS=0
N_CPUS=4

function cleanup() {
  rm -rf ${TEST_BASE}/data/*_0001.txt;
  rm -rf ${TEST_BASE}/envs;
  rm -rf ${TEST_BASE}/generators;
  rm -rf ${TEST_BASE}/logs;
}

cleanup

###########
# DATASET #
###########

DATASET_COMMON_ARGS="-b ${TEST_BASE} -k -d -mc ${MOL_COL} -sv -sif"
DATASET_FRAGMENT_ARGS="-fm ${FRAG_METHOD} -nf ${N_COMBINATIONS} -nf ${N_FRAGS}"

# pretraining data
echo "Test: Generate data for pretraining the fragment-based graph transformer..."
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_PRETRAINING} \
-o ${PRETRAINING_PREFIX} \
-mt graph ${DATASET_FRAGMENT_ARGS} \
-vf "${PRETRAINING_PREFIX}_${VOC_PREFIX}"
echo "Test: Done."

echo "Test: Generate data for pretraining the fragment-based sequence models..."
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_PRETRAINING} \
-o ${PRETRAINING_PREFIX} \
-mt smiles \
${DATASET_FRAGMENT_ARGS} \
-vf "${PRETRAINING_PREFIX}_${VOC_PREFIX}"
echo "Test: Done."

echo "Test: Generate data for pretraining the regular (no fragments) sequence model..."
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_PRETRAINING} \
-o ${PRETRAINING_PREFIX} \
-mt smiles \
-sm \
-nof \
-vf "${PRETRAINING_PREFIX}_${VOC_PREFIX}"
echo "Test: Done."

# finetuning data
echo "Test: Generate data for finetuning the fragment-based graph transformer..."
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_FINETUNING} \
-o ${FINETUNING_PREFIX} \
-mt graph \
${DATASET_FRAGMENT_ARGS} \
-vf "${FINETUNING_PREFIX}_${VOC_PREFIX}"
echo "Test: Done."

echo "Test: Generate data for pretraining the fragment-based sequence models..."
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_FINETUNING} \
-o ${FINETUNING_PREFIX} \
-mt smiles \
${DATASET_FRAGMENT_ARGS} \
-vf "${FINETUNING_PREFIX}_${VOC_PREFIX}"
echo "Test: Done."

echo "Test: Generate data for finetuning the regular (no fragments) sequence model..."
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_FINETUNING} \
-o ${FINETUNING_PREFIX} \
-sm \
-mt smiles \
-nof \
-vf "${FINETUNING_PREFIX}_${VOC_PREFIX}"
echo "Test: Done."

###############
# ENVIRONMENT #
###############
ENVIRON_COMMON_ARGS="-b ${TEST_BASE} -k -d"
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
TRAIN_COMMON_ARGS="-b ${TEST_BASE} -k -d -e ${TRAIN_EPOCHS} -bs ${TRAIN_BATCH} -gpu ${TRAIN_GPUS}"
TRAIN_VOCAB_ARGS="-vfs ${PRETRAINING_PREFIX}_${VOC_PREFIX} ${FINETUNING_PREFIX}_${VOC_PREFIX}"
#TRAIN_VOCAB_ARGS="" # uncomment to test with the default vocabularies

# pretraining

echo "Test: Pretrain fragment-based graph transformer..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${PRETRAINING_PREFIX}_${N_COMBINATIONS}:${N_FRAGS}_${FRAG_METHOD}" \
-o "${PRETRAINING_PREFIX}" \
-m PT \
-a graph
echo "Test: Done."

echo "Test: Pretrain fragment-based sequence encoder-decoder model..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${PRETRAINING_PREFIX}_${N_COMBINATIONS}:${N_FRAGS}_${FRAG_METHOD}" \
-o "${PRETRAINING_PREFIX}" \
-m PT \
-a ved
echo "Test: Done."

echo "Test: Pretrain fragment-based sequence encoder-decoder model with attention..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${PRETRAINING_PREFIX}_${N_COMBINATIONS}:${N_FRAGS}_${FRAG_METHOD}" \
-o "${PRETRAINING_PREFIX}" \
-m PT \
-a attn
echo "Test: Done."

echo "Test: Pretrain fragment-based sequence transformer model..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${PRETRAINING_PREFIX}_${N_COMBINATIONS}:${N_FRAGS}_${FRAG_METHOD}" \
-o "${PRETRAINING_PREFIX}" \
-m PT \
-a gpt
echo "Test: Done."

echo "Test: Pretrain regular (no fragments) single-network RNN model..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${PRETRAINING_PREFIX}_corpus" \
-o "${PRETRAINING_PREFIX}" \
-m PT \
-a rnn
echo "Test: Done."

# fine-tuning

echo "Test: Fine-tune fragment-based graph transformer..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${FINETUNING_PREFIX}_${N_COMBINATIONS}:${N_FRAGS}_${FRAG_METHOD}" \
-pt "${PRETRAINING_PREFIX}" \
-o "${FINETUNING_PREFIX}" \
-m FT \
-a graph
echo "Test: Done."

echo "Test: Fine-tune fragment-based vanilla sequence encoder-decoder..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${FINETUNING_PREFIX}_${N_COMBINATIONS}:${N_FRAGS}_${FRAG_METHOD}" \
-pt "${PRETRAINING_PREFIX}" \
-o "${FINETUNING_PREFIX}" \
-m \
FT \
-a ved
echo "Test: Done."

echo "Test: Fine-tune fragment-based sequence encoder-decoder with attention..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${FINETUNING_PREFIX}_${N_COMBINATIONS}:${N_FRAGS}_${FRAG_METHOD}" \
-pt "${PRETRAINING_PREFIX}" \
-o "${FINETUNING_PREFIX}" \
-m FT \
-a attn
echo "Test: Done."

echo "Test: Fine-tune fragment-based sequence transformer..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${FINETUNING_PREFIX}_${N_COMBINATIONS}:${N_FRAGS}_${FRAG_METHOD}" \
-pt "${PRETRAINING_PREFIX}" \
-o "${FINETUNING_PREFIX}" \
-m FT \
-a gpt
echo "Test: Done."

echo "Test: Fine-tune regular (no fragments) single-network RNN model..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${FINETUNING_PREFIX}_corpus" \
-pt "${PRETRAINING_PREFIX}" \
-o "${FINETUNING_PREFIX}" \
-m FT \
-a rnn
echo "Test: Done."

# reinforcement learning
RL_PREFIX="RL"
TARGET_ID="P29274"
ENVIRON_MODE="CLS"
ENVIRON_ALG="RF"
ENVIRON_THRESHOLD=6.5
TRAIN_RL_ARGS="-ta ${TARGET_ID} -et ${ENVIRON_MODE} -ea ${ENVIRON_ALG} -at ${ENVIRON_THRESHOLD}"

echo "Test: RL for the fragment-based graph transformer..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
${TRAIN_RL_ARGS} \
-i "${FINETUNING_PREFIX}_${N_COMBINATIONS}:${N_FRAGS}_${FRAG_METHOD}" \
-ag "${PRETRAINING_PREFIX}" \
-pr "${FINETUNING_PREFIX}" \
-o "${FINETUNING_PREFIX}_${RL}" \
-m RL \
-a graph
echo "Test: Done."

# FIXME: commented out for now, but should work at some point (issue: #10)
#echo "Test: RL for the fragment-based sequence encoder-decoder..."
#python train.py \
#${TRAIN_COMMON_ARGS} \
#${TRAIN_VOCAB_ARGS} \
#${TRAIN_RL_ARGS} \
#-i "${FINETUNING_PREFIX}_${N_COMBINATIONS}:${N_FRAGS}_${FRAG_METHOD}" \
#-ag "${PRETRAINING_PREFIX}" \
#-pr "${FINETUNING_PREFIX}" \
#-o "${FINETUNING_PREFIX}_${RL}" \
#-m RL \
#-a ved
#echo "Test: Done."

#echo "Test: RL for the fragment-based sequence encoder-decoder with attention..."
#python train.py \
#${TRAIN_COMMON_ARGS} \
#${TRAIN_VOCAB_ARGS} \
#${TRAIN_RL_ARGS} \
#-i "${FINETUNING_PREFIX}_${N_COMBINATIONS}:${N_FRAGS}_${FRAG_METHOD}" \
#-ag "${PRETRAINING_PREFIX}" \
#-pr "${FINETUNING_PREFIX}" \
#-o "${FINETUNING_PREFIX}_${RL}" \
#-m RL \
#-a attn
#echo "Test: Done."

echo "Test: RL for the fragment-based sequence transformer..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
${TRAIN_RL_ARGS} \
-i "${FINETUNING_PREFIX}_${N_COMBINATIONS}:${N_FRAGS}_${FRAG_METHOD}" \
-ag "${PRETRAINING_PREFIX}" \
-pr "${FINETUNING_PREFIX}" \
-o "${FINETUNING_PREFIX}_${RL}" \
-m RL \
-a gpt
echo "Test: Done."

echo "Test: RL for the regular (no fragments) single-network RNN model..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
${TRAIN_RL_ARGS} \
-i "${FINETUNING_PREFIX}_corpus" \
-ag "${PRETRAINING_PREFIX}" \
-pr "${FINETUNING_PREFIX}" \
-o "${FINETUNING_PREFIX}_${RL}" \
-m RL \
-a rnn
echo "Test: Done."

cleanup
