#!/bin/bash

EPOCHS=10
BATCH=16
LR=0.005
EMBEDDING_DIM=100                 #dimension of vectors
CLASSES=250                       #number of the category

TRAIN_DATA='./dataset/train.tsv'  #file for training dataset
TEST_DATA='./dataset/test.tsv'    #file for testing dataset


EMBEDDING_TYPE=1
EMBEDDING_PATH='/'                #file for pre-trained vectors
#0 is loading a pre-trained embedding, 1 is training from scratch(best case),
#2 is use random vectors but not update vectors(worst cases)
EXPERIMENT_NAME='best_case'
EXPERIMENT_LOG='./result/'$EXPERIMENT_NAME'.txt'

python main.py --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
--embedding_dim=$EMBEDDING_DIM --classes=$CLASSES --embedding_path=$EMBEDDING_PATH \
--train_data=$TRAIN_DATA --test_data=$TEST_DATA --embedding_type=$EMBEDDING_TYPE \
--experiment_name=$EXPERIMENT_NAME | tee $EXPERIMENT_LOG



EMBEDDING_TYPE=2
EMBEDDING_PATH='/'                #file for pre-trained vectors
#0 is loading a pre-trained embedding, 1 is training from scratch(best case),
#2 is use random vectors but not update vectors(worst cases)
EXPERIMENT_NAME='worst_case'
EXPERIMENT_LOG='./result/'$EXPERIMENT_NAME'.txt'

python main.py --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
--embedding_dim=$EMBEDDING_DIM --classes=$CLASSES --embedding_path=$EMBEDDING_PATH \
--train_data=$TRAIN_DATA --test_data=$TEST_DATA --embedding_type=$EMBEDDING_TYPE \
--experiment_name=$EXPERIMENT_NAME | tee $EXPERIMENT_LOG

