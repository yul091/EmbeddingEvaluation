# Project Description
 this repostory is designed for proposing a new 
 metric for evaluate the quality of embedding vectors

# File Structure

## `dataset` (directory contains the source dataset)

## `embedding_algorithms` (contains different algorithms to train the pre-trained embedding)

## `embedding_vec` (pre-trained vectors)

## `nlp_tasks` (directory for nlp related embedding)
+ `word_prediction`  
+ `automatic_summarisation`
+ `sentiment_analysis`
+ `machine_translation`

## `se_tasks` (directory for code related embedding )
+ `code_clone` (directory for task code clone detection)
+ `code_authorship` (directory for identify code authorship)
+ `code_completion` (directory for code completion)
+ `method_prediction` (directory for method name prediction)

## `vector_evaluation` (algorithms for metric evaluate the embedding)

# Suffix Description

## `.h5` store the pretrained model
## `.vec` the pre-trained vectors
## `.tsv` the daatset for downstream tasks