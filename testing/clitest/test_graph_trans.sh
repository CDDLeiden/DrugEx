#!/bin/bash

set -e

# pretraining data
echo $line
echo "Test: Generate data for pretraining the fragment-based graph transformer..."
echo $line
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_PRETRAINING} \
-o ${PRETRAINING_PREFIX} \
-mt graph ${DATASET_FRAGMENT_ARGS}
echo "Test: Done."

# finetuning data
echo $line
echo "Test: Generate data for finetuning the fragment-based graph transformer..."
echo $line
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_FINETUNING} \
-o ${FINETUNING_PREFIX} \
-mt graph \
${DATASET_FRAGMENT_ARGS}
echo "Test: Done."

# scaffold-based RL data
echo $line
echo "Test: Generate data for scaffold-based RL of the fragment-based graph models..."
echo $line
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_SCAFFOLD} \
-o ${SCAFFOLD_PREFIX} \
-mt graph \
-s \
${DATASET_FRAGMENT_ARGS}
echo "Test: Done."

echo $line
echo "Test: Generate data with specific fragment for finetuning the fragment-based graph models..."
echo $line
python -m drugex.dataset \
${DATASET_COMMON_ARGS} \
-i ${TEST_DATA_FINETUNING} \
-o ${FINETUNING_PREFIX}_pyrazine \
-mt graph \
-sf CCO \
${DATASET_FRAGMENT_ARGS}
echo "Test: Done."


# pretraining
echo $line
echo "Test: Pretrain fragment-based graph transformer..."
echo $line
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${PRETRAINING_PREFIX}" \
-o "${PRETRAINING_PREFIX}" \
-tm PT \
-mt graph \
-a trans
echo "Test: Done."

# finetuning
echo $line
echo "Test: Fine-tune fragment-based graph transformer..."
echo $line
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
-i "${FINETUNING_PREFIX}" \
-ag "${PRETRAINING_PREFIX}_graph_trans_PT" \
-o "${FINETUNING_PREFIX}" \
-tm FT \
-mt graph \
-a trans
echo "Test: Done."

# reinforcement learning
 echo $line
 echo "Test: RL for the fragment-based graph transformer..."
 echo $line
 python -m drugex.train \
 ${TRAIN_COMMON_ARGS} \
 ${TRAIN_VOCAB_ARGS} \
 ${TRAIN_RL_ARGS} \
 -i "${FINETUNING_PREFIX}" \
 -ag "${PRETRAINING_PREFIX}_graph_trans_PT" \
 -pr "${FINETUNING_PREFIX}_graph_trans_FT" \
 -o "${FINETUNING_PREFIX}_${RL_PREFIX}" \
 -tm RL \
 -mt graph \
 -a trans
 echo "Test: Done."

# scaffold-based RL
echo $line
echo "Test: scaffold-based RL for the fragment-based graph transformer..."
echo $line
python -m drugex.train \
${TRAIN_COMMON_ARGS} \
${TRAIN_VOCAB_ARGS} \
${TRAIN_RL_ARGS} \
-i "${SCAFFOLD_PREFIX}_graph.txt" \
-ag "${PRETRAINING_PREFIX}_graph_trans_PT" \
-pr "${FINETUNING_PREFIX}_graph_trans_FT" \
-o "${SCAFFOLD_PREFIX}_${RL_PREFIX}" \
-tm RL \
-mt graph \
-a trans \
-ns "${TRAIN_BATCH}" 
echo "Test: Done."

# generate
echo $line
echo "Test: Generate molecules with graph transformer ..."
echo $line
python -m drugex.generate \
${DESIGN_COMMON_ARGS} \
${TRAIN_RL_ARGS} \
-i "${FINETUNING_PREFIX}" \
-g "${FINETUNING_PREFIX}_${RL_PREFIX}_graph_trans_RL" \
-vfs "${FINETUNING_PREFIX}" \
--keep_invalid
echo "Test: Done."