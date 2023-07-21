#!/bin/bash

RES_DIR='se_tasks/code_search/result'
if [ ! -d $RES_DIR ]; then
  mkdir $RES_DIR
else
  echo dir exist
fi


BATCH=512
LR=0.005
DATA_PATH='se_tasks/code_search/dataset/example/'
# TRAIN_DATA='se_tasks/code_search/dataset/train.tsv'
# TEST_DATA='se_tasks/code_search/dataset/test.tsv'


# EMBED_TYPE=1
# EMBEDDING_DIM=100                 #dimension of vectors
# EMBEDDING_PATH='/'                #file for pre-trained vectors
# EXPERIMENT_NAME='best_case'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=0 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR --data_path=$DATA_PATH \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG



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


# EMBED_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/Doc2VecEmbedding0.vec'
# EXPERIMENT_NAME='doc2vec_cowb'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=2 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR --data_path=$DATA_PATH \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG


# EMBED_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/Doc2VecEmbedding1.vec'
# EXPERIMENT_NAME='doc2vec_skipgram'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=1 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR --data_path=$DATA_PATH \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG


# EMBED_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/Word2VecEmbedding0.vec'
# EXPERIMENT_NAME='word2vec_cowb'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=3 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR --data_path=$DATA_PATH \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG


# EMBED_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/Word2VecEmbedding1.vec'
# EXPERIMENT_NAME='word2vec_skipgram'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=3 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR --data_path=$DATA_PATH \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG



# EMBED_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/FastEmbeddingcbow.vec'
# EXPERIMENT_NAME='fasttext_cowb'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=0 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR --data_path=$DATA_PATH \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG


# EMBED_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/FastEmbeddingskipgram.vec'
# EXPERIMENT_NAME='fasttext_skipgram'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=0 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR --data_path=$DATA_PATH \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG


# EMBED_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/GloVeEmbeddingNone.vec'
# EXPERIMENT_NAME='glove'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=4 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR --data_path=$DATA_PATH \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG



# EMBED_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/ori_code2seq.vec'
# EXPERIMENT_NAME='code2seq'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=4 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR --data_path=$DATA_PATH \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG


# EMBED_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/code2vec.vec'
# EXPERIMENT_NAME='code2vec'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=4 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR --data_path=$DATA_PATH \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG