#! /bin/bash

# # NLI distillation reranker topiocqa 
# # train_file_path="data/topiocqa/merge_entail/train_bm25_hard_negs_topi_new_entail.json"  # without correction
# # train_file_path="data/topiocqa/merge_entail/train_bm25_hard_negs_topi_new_entail_minmaxdot_v11.json"
# train_file_path="data/qrecc/new_data/nli/train_with_negs_entail_minmaxdot_corr_09_07.json"
# model_output_path="experiments/topi_reranker_0907/output"
# log_path="experiments/topi_reranker_0907/log/train_logging.log"
# reranker_encoder_path="deepset/roberta-base-squad2"
# # reranker_encoder_path="3ricL/ad-hoc-ance-msmarco"

# mkdir -p $model_output_path

# NCCL_P2P_DISABLE=1 MASTER_ADDR=localhost MASTER_PORT=30000 CUDA_VISIBLE_DEVICES=0,1,2 nohup torchrun --nproc_per_node=3 \
#       --rdzv_backend=c10d --rdzv_endpoint=localhost:30000 --standalone \
#       src/rerank/train_reranker_class.py \
#       --reranker_encoder_path=$reranker_encoder_path \
#       --train_file_path=$train_file_path \
#       --output_dir=$model_output_path \
#       --log_path=$log_path \
#       --train_batch_size=8 \
#       --num_train_epochs=6 \
#       --max_query_length=32 \
#       --max_doc_length=384 \
#       --max_response_length=64 \
#       --max_concat_length=512 \
#       --dataset="topiocqa" \
#       --lambda_nli=2 \
#       --n_gpu=3 \
#       --binary_cls \
#       --num_hard_negatives=4 > experiments/topi_reranker_0907/log/train_nohup.out 2>&1 &

# # train_file_path="data/qrecc/new_data/nli/train_with_negs_entail.json"
# train_file_path="data/qrecc/new_data/nli/train_with_negs_entail_minmaxdot_corr.json"
# model_output_path="experiments/qrecc_reranker/output"
# log_path="experiments/qrecc_reranker/log/train_logging.log"
# reranker_encoder_path="deepset/roberta-base-squad2"
# # reranker_encoder_path="3ricL/ad-hoc-ance-msmarco"

# mkdir -p $model_output_path

# NCCL_P2P_DISABLE=1 MASTER_ADDR=localhost MASTER_PORT=30000 CUDA_VISIBLE_DEVICES=5,6,7 nohup torchrun --nproc_per_node=3 \
#       --rdzv_backend=c10d --rdzv_endpoint=localhost:30000 --standalone \
#       src/rerank/train_reranker.py \
#       --reranker_encoder_path=$reranker_encoder_path \
#       --train_file_path=$train_file_path \
#       --output_dir=$model_output_path \
#       --log_path=$log_path \
#       --train_batch_size=8 \
#       --num_train_epochs=2 \
#       --max_query_length=32 \
#       --max_doc_length=384 \
#       --max_response_length=64 \
#       --max_concat_length=512 \
#       --dataset="qrecc" \
#       --lambda_nli=2 \
#       --n_gpu=3 \
#       --bm25_per_sample 2 --topic_per_sample 2 --random_per_sample 3 \
#       --num_hard_negatives=4 > experiments/qrecc_reranker/log/train_nohup.out 2>&1 &





# train_file_path="data/topiocqa/merge_entail/train_bm25_hard_negs_topi_new_entail.json"  # without correction
train_file_path="data/topiocqa/merge_entail/train_bm25_hard_negs_topi_new_entail_minmaxdot_v11.json"
model_output_path="experiments/topi_reranker/output"
log_path="experiments/topi_reranker/log/train_logging.log"
reranker_encoder_path="deepset/roberta-base-squad2"

mkdir -p $model_output_path

NCCL_P2P_DISABLE=1 MASTER_ADDR=localhost MASTER_PORT=30000 CUDA_VISIBLE_DEVICES=6,7 nohup torchrun --nproc_per_node=2 \
      --rdzv_backend=c10d --rdzv_endpoint=localhost:30000 --standalone \
      src/rerank/train_reranker.py \
      --reranker_encoder_path=$reranker_encoder_path \
      --train_file_path=$train_file_path \
      --output_dir=$model_output_path \
      --log_path=$log_path \
      --train_batch_size=8 \
      --num_train_epochs=6 \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=64 \
      --max_concat_length=512 \
      --dataset="topiocqa" \
      --lambda_nli=2 \
      --n_gpu=2 \
      --bm25_per_sample 2 --topic_per_sample 2 --random_per_sample 2 \
      --num_hard_negatives=4 > experiments/topi_reranker/log/train_nohup.out 2>&1 &
