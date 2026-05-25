#! /bin/bash




# CUDA_VISIBLE_DEVICES=5 nohup python src/train_QRACDR.py \
#     --pretrained_query_encoder_path="3ricL/ad-hoc-ance-msmarco" \
#     --pretrained_oracle_encoder_path="3ricL/ad-hoc-ance-msmarco" \
#     --train_file_path=$train_file_path \
#     --log_dir_path=$log_dir_path \
#     --model_output_path=$model_output_path \
#     --log_path=$log_path \
#     --per_gpu_train_batch_size=64 \
#     --num_train_epochs=10 \
#     --max_query_length=32 \
#     --max_doc_length=384 \
#     --max_response_length=64 \
#     --max_concat_length=512 \
#     --dataset="topiocqa" \
#     --mode="mse+CL" \
#     --n_gpu=1 > experiments/base_topiocqa/log/train_nohup.out 2>&1 &



# train_file_path="data/topiocqa/train_with_rewrite_bm25.json"
# log_dir_path="experiments/base_ance_cls/log/train_summary"
# model_output_path="experiments/base_ance_cls/output"
# log_path="experiments/base_ance_cls/log/train_logging_new.log"
# ance_model="3ricL/ad-hoc-ance-msmarco"


# NCCL_P2P_DISABLE=1 MASTER_ADDR=localhost MASTER_PORT=30000 CUDA_VISIBLE_DEVICES=4 nohup torchrun --nproc_per_node=1 \
#       --rdzv_backend=c10d --rdzv_endpoint=localhost:30000 --standalone \
#       src/train_QRACDR.py \
#       --pretrained_query_encoder_path=$ance_model \
#       --pretrained_oracle_encoder_path=$ance_model \
#       --train_file_path=$train_file_path \
#       --log_dir_path=$log_dir_path \
#       --model_output_path=$model_output_path \
#       --log_path=$log_path \
#       --per_gpu_train_batch_size=32 \
#       --num_train_epochs=20 \
#       --max_query_length=32 \
#       --max_doc_length=384 \
#       --max_response_length=64 \
#       --max_concat_length=512 \
#       --dataset="topiocqa" \
#       --mode="add_CL" \
#       --n_gpu=1 > experiments/base_ance_cls/log/train_nohup_new.out 2>&1 &



# traindialogue-query rewrite distillation
train_file_path="data/qrecc/new_data/train_with_negs.json"
log_dir_path="experiments/qrecc_ance/qrecc_dense_ance/log/train_summary"
model_output_path="experiments/qrecc_ance/qrecc_dense_ance/output"
log_path="experiments/qrecc_ance/qrecc_dense_ance/log/train_logging.log"
ance_model="3ricL/ad-hoc-ance-msmarco"


NCCL_P2P_DISABLE=1 MASTER_ADDR=localhost MASTER_PORT=30000 CUDA_VISIBLE_DEVICES=4 nohup torchrun --nproc_per_node=1 \
      --rdzv_backend=c10d --rdzv_endpoint=localhost:30000 --standalone \
      src/retrieval/train_ance.py \
      --pretrained_query_encoder_path=$ance_model \
      --pretrained_oracle_encoder_path=$ance_model \
      --train_file_path=$train_file_path \
      --log_dir_path=$log_dir_path \
      --model_output_path=$model_output_path \
      --log_path=$log_path \
      --per_gpu_train_batch_size=32 \
      --num_train_epochs=20 \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=64 \
      --max_concat_length=512 \
      --dataset="qrecc" \
      --dense \
      --n_gpu=1 > experiments/qrecc_ance/qrecc_dense_ance/log/train_nohup.out 2>&1 &

# --conv2query \