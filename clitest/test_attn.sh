#!/bin/bash

set -e

echo "Test: Generate data for pretraining the fragment-based sequence models..."
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_PRETRAINING} \
-o ${PRETRAINING_PREFIX} \
-mt smiles \
${DATASET_FRAGMENT_ARGS}
echo "Test: Done."

echo "Test: Generate data for pretraining the fragment-based sequence models..."
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_FINETUNING} \
-o ${FINETUNING_PREFIX} \
-mt smiles \
${DATASET_FRAGMENT_ARGS}
echo "Test: Done."

echo "Test: Pretrain fragment-based sequence encoder-decoder model with attention..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${PRETRAINING_PREFIX}" \
-o "${PRETRAINING_PREFIX}" \
-m PT \
-a attn \
-mt smiles
echo "Test: Done."

echo "Test: Fine-tune fragment-based sequence encoder-decoder with attention..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${FINETUNING_PREFIX}" \
-pt "${PRETRAINING_PREFIX}" \
-o "${FINETUNING_PREFIX}" \
-m FT \
-a attn \
-mt smiles
echo "Test: Done."

# FIXME: commented out for now, but should work at some point (issue: #10)
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