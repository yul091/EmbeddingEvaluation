#!/bin/bash

RES_DIR='se_tasks/code_summary/result'
if [ ! -d $RES_DIR ]; then
  mkdir $RES_DIR
else
  echo dir exist
fi


EPOCHS=50
BATCH=128
LR=0.005

TK_PATH='dataset/java-small-preprocess/tk.pkl'
TRAIN_DATA='dataset/java-small-preprocess/train.pkl'  #file for training dataset
TEST_DATA='dataset/java-small-preprocess/test.pkl'    #file for testing dataset



# EMBEDDING_TYPE=1
# EMBEDDING_DIM=100                 #dimension of vectors
# EMBEDDING_PATH='/'                #file for pre-trained vectors
# EXPERIMENT_NAME='best_case'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=3 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG



EMBEDDING_TYPE=2
EMBEDDING_DIM=100
EMBEDDING_PATH='/'
EXPERIMENT_NAME='worst_case'
EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
echo $EXPERIMENT_NAME
CUDA_VISIBLE_DEVICES=0 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
--embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
--train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
--experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG



# EMBEDDING_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/Doc2VecEmbedding0.vec'
# EXPERIMENT_NAME='doc2vec_cowb'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=3 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG

# EMBEDDING_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/Doc2VecEmbedding1.vec'
# EXPERIMENT_NAME='doc2vec_skipgram'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=0 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG


# EMBEDDING_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/Word2VecEmbedding0.vec'
# EXPERIMENT_NAME='word2vec_cowb'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=2 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG


# EMBEDDING_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/Word2VecEmbedding1.vec'
# EXPERIMENT_NAME='word2vec_skipgram'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=4 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG


# EMBEDDING_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/FastEmbeddingcbow.vec'
# EXPERIMENT_NAME='fasttext_cowb'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=5 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG


# EMBEDDING_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/FastEmbeddingskipgram.vec'
# EXPERIMENT_NAME='fasttext_skipgram'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=6 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG



# EMBEDDING_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/GloVeEmbeddingNone.vec'
# EXPERIMENT_NAME='glove'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=7 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG


# EMBEDDING_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/ori_code2seq.vec'
# EXPERIMENT_NAME='code2seq'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=0 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG


# EMBEDDING_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec/100_2/code2vec.vec'
# EXPERIMENT_NAME='code2vec'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=5 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG