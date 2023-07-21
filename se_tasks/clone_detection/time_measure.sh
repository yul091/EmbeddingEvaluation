#!/bin/bash

RES_DIR='se_tasks/clone_detection/time'
if [ ! -d $RES_DIR ]; then
  mkdir $RES_DIR
else
  echo dir exist
fi

EPOCHS=3
BATCH=512
LR=0.005
TRAIN_DATA='se_tasks/clone_detection/data/train.pkl'
TEST_DATA='se_tasks/clone_detection/data/test.pkl'


EMBEDDING_TYPE=2
EMBEDDING_DIM=100
EMBEDDING_PATH='/'
EXPERIMENT_NAME='worst_case'
EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
echo $EXPERIMENT_NAME
CUDA_VISIBLE_DEVICES=0 python -m se_tasks.clone_detection.scripts.main \
--train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
--epochs=$EPOCHS --batch=$BATCH --lr=$LR \
--embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
--experiment_name=$EXPERIMENT_NAME


EMBEDDING_TYPE=1
EMBEDDING_DIM=100
EMBEDDING_PATH='/'
EXPERIMENT_NAME='best_case'
EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
echo $EXPERIMENT_NAME
CUDA_VISIBLE_DEVICES=0 python -m se_tasks.clone_detection.scripts.main \
--train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
--epochs=$EPOCHS --batch=$BATCH --lr=$LR \
--embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
--experiment_name=$EXPERIMENT_NAME