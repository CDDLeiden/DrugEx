#!/bin/bash

set -e

# Pretraining ###############################################################
echo $line
echo "Test: Generate data for pretraining the regular (no fragments) sequence model..."
echo $line
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_PRETRAINING} \
-o ${PRETRAINING_PREFIX} \
-mt smiles \
-vf 'voc_smiles.txt' \
-nof
echo "Test: Done."

echo $line
echo "Test: Pretrain regular (no fragments) single-network RNN model..."
echo $line
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${PRETRAINING_PREFIX}" \
-o "${PRETRAINING_PREFIX}" \
-vfs "${PRETRAINING_PREFIX}_corpus.txt.vocab" \
-m PT \
-a rnn \
-mt smiles
echo "Test: Done."

echo $line
echo "Test: Pretrain regular (no fragments) single-network RNN model with GRUs..."
echo $line
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${PRETRAINING_PREFIX}" \
-o "${PRETRAINING_PREFIX}_${GRU_PREFIX}" \
-vfs "${PRETRAINING_PREFIX}_corpus.txt.vocab" \
-m PT \
-a rnn \
-gru \
-mt smiles
echo "Test: Done."

# Finetuning ###############################################################
echo $line
echo "Test: Generate data for finetuning the regular (no fragments) sequence model..."
echo $line
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_FINETUNING} \
-o ${FINETUNING_PREFIX} \
-mt smiles \
-vf 'voc_smiles.txt' \
-nof
echo "Test: Done."

echo $line
echo "Test: Fine-tune regular (no fragments) single-network RNN model..."
echo $line
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${FINETUNING_PREFIX}" \
-ag "${PRETRAINING_PREFIX}_smiles_rnn_PT" \
-o "${FINETUNING_PREFIX}" \
-vfs "${PRETRAINING_PREFIX}_corpus.txt.vocab" \
-m FT \
-a rnn \
-mt smiles
echo "Test: Done."

echo $line
echo "Test: Fine-tune regular (no fragments) single-network RNN model with GRUs..."
echo $line
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${FINETUNING_PREFIX}" \
-ag "${PRETRAINING_PREFIX}_${GRU_PREFIX}_smiles_rnn_PT" \
-o "${FINETUNING_PREFIX}_${GRU_PREFIX}" \
-vfs "${PRETRAINING_PREFIX}_corpus.txt.vocab" \
-m FT \
-a rnn \
-gru \
-mt smiles
echo "Test: Done."

# Reinforcement Learning ###############################################################
echo $line
echo "Test: RL for the regular (no fragments) single-network RNN model..."
echo $line
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
${TRAIN_RL_ARGS} \
-i "${FINETUNING_PREFIX}" \
-ag "${PRETRAINING_PREFIX}_smiles_rnn_PT" \
-pr "${FINETUNING_PREFIX}_smiles_rnn_FT" \
-o "${FINETUNING_PREFIX}_${RL_PREFIX}" \
-vfs "${PRETRAINING_PREFIX}_corpus.txt.vocab" \
-m RL \
-a rnn \
-mt smiles
echo "Test: Done."

echo $line
echo "Test: RL for the regular (no fragments) single-network RNN model with GRUs..."
echo $line
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
${TRAIN_RL_ARGS} \
-i "${FINETUNING_PREFIX}" \
-ag "${PRETRAINING_PREFIX}_${GRU_PREFIX}_smiles_rnn_PT" \
-pr "${FINETUNING_PREFIX}_${GRU_PREFIX}_smiles_rnn_FT" \
-o "${FINETUNING_PREFIX}_${GRU_PREFIX}_${RL_PREFIX}" \
-m RL \
-a rnn \
-gru \
-mt smiles 
echo "Test: Done."

# Designer ###############################################################
echo $line
echo "Test: Generate molecules with sequence RNN ..."
echo $line
python -m drugex.designer \
${DESIGN_COMMON_ARGS} \
-i "${FINETUNING_PREFIX}" \
-g "${FINETUNING_PREFIX}_${RL_PREFIX}_smiles_rnn_RL" \
-vfs "${FINETUNING_PREFIX}" \
--keep_invalid
echo "Test: Done."