#!/bin/bash

set -e

# pretraining data
echo "Test: Generate data for pretraining the fragment-based graph transformer..."
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_PRETRAINING} \
-o ${PRETRAINING_PREFIX} \
-mt graph ${DATASET_FRAGMENT_ARGS}
echo "Test: Done."

# finetuning data
echo "Test: Generate data for finetuning the fragment-based graph transformer..."
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_FINETUNING} \
-o ${FINETUNING_PREFIX} \
-mt graph \
${DATASET_FRAGMENT_ARGS}
echo "Test: Done."

# pretraining

echo "Test: Pretrain fragment-based graph transformer..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${PRETRAINING_PREFIX}" \
-o "${PRETRAINING_PREFIX}" \
-m PT \
-a graph
echo "Test: Done."

# finetuning

echo "Test: Fine-tune fragment-based graph transformer..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${FINETUNING_PREFIX}" \
-pt "${PRETRAINING_PREFIX}" \
-o "${FINETUNING_PREFIX}" \
-m FT \
-a graph
echo "Test: Done."

# reinforcement learning

echo "Test: RL for the fragment-based graph transformer..."
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
${TRAIN_RL_ARGS} \
-i "${FINETUNING_PREFIX}" \
-ag "${PRETRAINING_PREFIX}_graph_graph_PT" \
-pr "${FINETUNING_PREFIX}_graph_graph_FT" \
-o "${FINETUNING_PREFIX}_${RL}" \
-m RL \
-a graph
echo "Test: Done."

