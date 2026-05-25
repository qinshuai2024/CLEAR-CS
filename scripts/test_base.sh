#! /bin/bash

# # qrecc
# # trained_model_path="experiments/base/-best-model"
# trained_model_path="3ricL/ad-hoc-ance-msmarco"
# test_file_path="data/qrecc/new_preprocessed/test.json"
# passage_embeddings_dir_path="data/qrecc/new_preprocessed/embeds"
# passage_offset2pid_path="data/qrecc/new_preprocessed/tokenized/offset2pid.pickle"
# qrel_output_path="experiments/base_ance/qrecc/test"
# output_trec_file="qrecc_trec_ance_top10"
# trec_gold_qrel_file_path="data/qrecc/new_preprocessed/qrecc_qrel.tsv"
# output_high_topk_path="experiments/base_ance/qrecc/data/test_high_top_100_ance_rq.json"

# CUDA_VISIBLE_DEVICES=0 nohup python src/test_QRACDR.py --pretrained_encoder_path=$trained_model_path \
#     --test_file_path=$test_file_path \
#     --passage_embeddings_dir_path=$passage_embeddings_dir_path \
#     --passage_offset2pid_path=$passage_offset2pid_path \
#     --qrel_output_path=$qrel_output_path \
#     --output_trec_file=$output_trec_file \
#     --trec_gold_qrel_file_path=$trec_gold_qrel_file_path \
#     --output_high_topk_path=$output_high_topk_path \
#     --test_type="convqa" \
#     --max_query_length=32 \
#     --max_doc_length=384 \
#     --max_response_length=64 \
#     --max_concat_length=512 \
#     --dataset="qrecc" \
#     --passage_block_num=22 \
#     --top_k=100 \
#     --per_gpu_test_batch_size=8 > experiments/base_ance/qrecc/log/test_nohup_top10_ance_rq.out 2>&1 &


# # Predict on training dataset
# trained_model_path="3ricL/ad-hoc-ance-msmarco"
# test_file_path="data/qrecc/new_preprocessed/train.json"
# passage_embeddings_dir_path="data/qrecc/new_preprocessed/embeds"
# passage_offset2pid_path="data/qrecc/new_preprocessed/tokenized/offset2pid.pickle"
# qrel_output_path="experiments/base_ance/qrecc/test"
# output_trec_file="qrecc_train_trec_best"
# trec_gold_qrel_file_path="data/qrecc/new_preprocessed/qrecc_qrel_train.tsv"
# output_high_topk_path="experiments/base_ance/qrecc/data/high_top_100.json"

# CUDA_VISIBLE_DEVICES=4 nohup python src/test_QRACDR.py --pretrained_encoder_path=$trained_model_path \
#     --test_file_path=$test_file_path \
#     --passage_embeddings_dir_path=$passage_embeddings_dir_path \
#     --passage_offset2pid_path=$passage_offset2pid_path \
#     --qrel_output_path=$qrel_output_path \
#     --output_trec_file=$output_trec_file \
#     --trec_gold_qrel_file_path=$trec_gold_qrel_file_path \
#     --output_high_topk_path=$output_high_topk_path \
#     --test_type="convqa" \
#     --max_query_length=32 \
#     --max_doc_length=384 \
#     --max_response_length=64 \
#     --max_concat_length=512 \
#     --dataset="qrecc" \
#     --passage_block_num=22 \
#     --per_gpu_test_batch_size=8 > experiments/base_ance/qrecc/log/test_nohup_train.out 2>&1 &


# # topiocqa
# ance_model="3ricL/ad-hoc-ance-msmarco"
# # trained_model_path="3ricL/ad-hoc-ance-msmarco"
# test_file_path="data/topiocqa/test_with_rewrite.json"
# passage_embeddings_dir_path="data/topiocqa/emb_title/embeddings"
# passage_offset2pid_path="data/topiocqa/emb_title/tokenized/offset2pid.pickle"
# qrel_output_path="experiments/base_ance_cls/test"
# output_trec_file="topiocqa_trec_best_cls_16"
# trec_gold_qrel_file_path="data/topiocqa/topiocqa_qrel.tsv"
# output_high_topk_path="experiments/base_ance_cls/data/test_high_top_100_base_17.json"

# CUDA_VISIBLE_DEVICES=3 nohup python src/retrieval/test_ance.py --pretrained_encoder_path=$ance_model \
#     --test_file_path=$test_file_path \
#     --passage_embeddings_dir_path=$passage_embeddings_dir_path \
#     --passage_offset2pid_path=$passage_offset2pid_path \
#     --qrel_output_path=$qrel_output_path \
#     --output_trec_file=$output_trec_file \
#     --trec_gold_qrel_file_path=$trec_gold_qrel_file_path \
#     --output_high_topk_path=$output_high_topk_path \
#     --test_type="convqa" \
#     --max_query_length=32 \
#     --max_doc_length=384 \
#     --max_response_length=64 \
#     --max_concat_length=512 \
#     --dataset="topiocqa" \
#     --passage_block_num=26 \
#     --top_k=100 \
#     --per_gpu_test_batch_size=8 > experiments/topi_reranker_0907/log/test_nohup_top100_base_17.out 2>&1 &

# # predicttrain集
# # # trained_model_path="3ricL/ad-hoc-ance-msmarco"
# trained_model_path="experiments/base_topiocqa/output/0-model-epoch13-step6636-loss0.0281"  # qracdr
# # trained_model_path="experiments/base_ance_cls/output/0-model-epoch17-step25578-loss0.0008"
# test_file_path="data/topiocqa/train_with_rewrite.json"
# passage_embeddings_dir_path="data/topiocqa/emb_title/embeddings"
# passage_offset2pid_path="data/topiocqa/emb_title/tokenized/offset2pid.pickle"
# qrel_output_path="experiments/base_topiocqa/test"
# output_trec_file="topiocqa_trec_train"
# trec_gold_qrel_file_path="data/topiocqa/train_gold.trec"
# output_high_topk_path="experiments/base_topiocqa/data/train_high_top_10_qracdr.json"

# CUDA_VISIBLE_DEVICES=3 nohup python src/test_QRACDR.py --pretrained_encoder_path=$trained_model_path \
#     --test_file_path=$test_file_path \
#     --passage_embeddings_dir_path=$passage_embeddings_dir_path \
#     --passage_offset2pid_path=$passage_offset2pid_path \
#     --qrel_output_path=$qrel_output_path \
#     --output_trec_file=$output_trec_file \
#     --trec_gold_qrel_file_path=$trec_gold_qrel_file_path \
#     --output_high_topk_path=$output_high_topk_path \
#     --test_type="convqa" \
#     --max_query_length=32 \
#     --max_doc_length=384 \
#     --max_response_length=64 \
#     --max_concat_length=512 \
#     --dataset="topiocqa" \
#     --passage_block_num=26 \
#     --top_k=10 \
#     --per_gpu_test_batch_size=8 > experiments/base_topiocqa/log/train_nohup_top10_qracdr.out 2>&1 &


# cast19 / 20
trained_model_path="experiments/base_topiocqa/output/0-model-epoch16-step8058-loss0.0409"
# trained_model_path="3ricL/ad-hoc-ance-msmarco"
# trained_model_path="experiments/base/-best-model"
test_file_path="data/cast/data_cast19/new_data/cast19_test_topiocqa.jsonl"
# test_file_path="data/cast/data_cast20/new_data/cast20_test_topiocqa.jsonl"
passage_embeddings_dir_path="data/cast/cast2019_embeddings"
passage_offset2pid_path="data/cast/cast2019_tokenized/offset2pid.pickle"
qrel_output_path="experiments/cast/cast19/test"
output_trec_file="topiocqa_trec_best_cls_16"
trec_gold_qrel_file_path="data/cast/data_cast19/new_data/cast19_qrel.tsv"
# trec_gold_qrel_file_path="data/cast/data_cast20/new_data/cast20_qrel.tsv"
output_high_topk_path="experiments/cast/cast19/test_qracdr_topi.json"

CUDA_VISIBLE_DEVICES=0 nohup python src/retrieval/test_ance.py --pretrained_encoder_path=$trained_model_path \
    --test_file_path=$test_file_path \
    --passage_embeddings_dir_path=$passage_embeddings_dir_path \
    --passage_offset2pid_path=$passage_offset2pid_path \
    --qrel_output_path=$qrel_output_path \
    --output_trec_file=$output_trec_file \
    --trec_gold_qrel_file_path=$trec_gold_qrel_file_path \
    --output_high_topk_path=$output_high_topk_path \
    --test_type="convqa" \
    --max_query_length=32 \
    --max_doc_length=384 \
    --max_response_length=64 \
    --max_concat_length=512 \
    --dataset="topiocqa" \
    --passage_block_num=26 \
    --top_k=100 \
    --per_gpu_test_batch_size=8 > experiments/cast/cast19/log/test_qracdr_topi.out 2>&1 &
