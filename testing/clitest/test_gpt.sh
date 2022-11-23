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

echo "Test: Pretrain fragment-based sequence transformer model..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${PRETRAINING_PREFIX}" \
-o "${PRETRAINING_PREFIX}" \
-m PT \
-a trans \
-mt smiles
echo "Test: Done."

echo "Test: Fine-tune fragment-based sequence transformer..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${FINETUNING_PREFIX}" \
-pt "${PRETRAINING_PREFIX}" \
-o "${FINETUNING_PREFIX}" \
-m FT \
-a trans \
-mt smiles
echo "Test: Done."

echo "Test: RL for the fragment-based sequence transformer..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
${TRAIN_RL_ARGS} \
-i "${FINETUNING_PREFIX}" \
-ag "${PRETRAINING_PREFIX}_smiles_trans_PT" \
-pr "${FINETUNING_PREFIX}_smiles_trans_FT" \
-o "${FINETUNING_PREFIX}_${RL}" \
-m RL \
-a trans \
-mt smiles
echo "Test: Done."