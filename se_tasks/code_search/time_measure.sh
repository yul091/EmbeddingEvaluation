#!/bin/bash

RES_DIR='se_tasks/code_search/result'
if [ ! -d $RES_DIR ]; then
  mkdir $RES_DIR
else
  echo dir exist
fi


BATCH=256
LR=0.005
DATA_PATH='se_tasks/code_search/dataset/example/'



EMBED_TYPE=2
EMBEDDING_DIM=100
EMBEDDING_PATH='/'
EXPERIMENT_NAME='worst_case'
EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
echo $EXPERIMENT_NAME
CUDA_VISIBLE_DEVICES=1 python -m se_tasks.code_search.scripts.train \
--embed_type=$EMBED_TYPE --learning_rate=$LR --data_path=$DATA_PATH \
--embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
--experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG


EMBED_TYPE=1
EMBEDDING_DIM=100
EMBEDDING_PATH='/'
EXPERIMENT_NAME='best_case'
EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
echo $EXPERIMENT_NAME
CUDA_VISIBLE_DEVICES=1 python -m se_tasks.code_search.scripts.train \
--embed_type=$EMBED_TYPE --learning_rate=$LR --data_path=$DATA_PATH \
--embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
--experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG