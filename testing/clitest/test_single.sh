#!/bin/bash

set -e

echo "Test: Generate data for pretraining the regular (no fragments) sequence model..."
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_PRETRAINING} \
-o ${PRETRAINING_PREFIX} \
-mt smiles \
-nof
echo "Test: Done."

echo "Test: Generate data for finetuning the regular (no fragments) sequence model..."
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_FINETUNING} \
-o ${FINETUNING_PREFIX} \
-mt smi \
-nof
echo "Test: Done."

echo "Test: Pretrain regular (no fragments) single-network RNN model..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${PRETRAINING_PREFIX}" \
-o "${PRETRAINING_PREFIX}" \
-m PT \
-a rnn \
-mt smiles
echo "Test: Done."

echo "Test: Fine-tune regular (no fragments) single-network RNN model..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${FINETUNING_PREFIX}" \
-pt "${PRETRAINING_PREFIX}" \
-o "${FINETUNING_PREFIX}" \
-m FT \
-a rnn \
-mt smiles
echo "Test: Done."

echo "Test: RL for the regular (no fragments) single-network RNN model..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
${TRAIN_RL_ARGS} \
-i "${FINETUNING_PREFIX}" \
-ag "${PRETRAINING_PREFIX}_smiles_rnn_PT" \
-pr "${FINETUNING_PREFIX}_smiles_rnn_FT" \
-o "${FINETUNING_PREFIX}_${RL}" \
-m RL \
-a rnn \
-mt smiles
echo "Test: Done."