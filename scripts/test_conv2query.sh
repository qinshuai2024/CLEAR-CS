# qrecc
# trained_model_path="3ricL/ad-hoc-ance-msmarco"
# # trained_model_path="experiments/base/-best-model"
# test_file_path="data/qrecc/new_preprocessed/test.json"
# # new_query_jsonl="data/qrecc/new_preprocessed/LLM_retrieval/generated_queries.jsonl"
# new_query_jsonl="data/qrecc/new_preprocessed/LLM_retrieval/generated_queries_rq_all.jsonl"
# output_trec_file="experiments/base/LLM_retrieval/test/qrecc_trec_llm_rq_all"
# trec_gold_qrel_file_path="data/qrecc/new_preprocessed/qrecc_qrel.tsv"
# output_high_topk_path="experiments/base/LLM_retrieval/test/high_topk_rq_all.json"

# # CUDA_VISIBLE_DEVICES=0 
# CUDA_VISIBLE_DEVICES=4  nohup python src/LLM_retrieval/test_query2query.py --pretrained_encoder_path=$trained_model_path \
#     --test_file_path=$test_file_path \
#     --new_query_jsonl=$new_query_jsonl \
#     --output_trec_file=$output_trec_file \
#     --trec_gold_qrel_file_path=$trec_gold_qrel_file_path \
#     --output_high_topk_path=$output_high_topk_path \
#     --max_concat_length=512 \
#     --top_k=100 \
#     --batch_size=512 > experiments/base/LLM_retrieval/log/test_nohup_rq_all.out 2>&1 &



# # Predict on training dataset
# trained_model_path="experiments/base/-best-model"
# test_file_path="data/qrecc/new_preprocessed/train.json"
# new_query_jsonl="data/qrecc/new_preprocessed/LLM_retrieval/train_split/train_generated_queries.jsonl"
# output_trec_file="experiments/base/LLM_retrieval/test/qrecc_trec_llm_train_val"
# trec_gold_qrel_file_path="data/qrecc/new_preprocessed/qrecc_qrel_train.tsv"
# output_high_topk_path="experiments/base/LLM_retrieval/test/high_topk_train_val_rq.json"

# # CUDA_VISIBLE_DEVICES=0 
# CUDA_VISIBLE_DEVICES=4  nohup python src/LLM_retrieval/test_query2query.py --pretrained_encoder_path=$trained_model_path \
#     --test_file_path=$test_file_path \
#     --new_query_jsonl=$new_query_jsonl \
#     --output_trec_file=$output_trec_file \
#     --trec_gold_qrel_file_path=$trec_gold_qrel_file_path \
#     --output_high_topk_path=$output_high_topk_path \
#     --max_concat_length=512 \
#     --top_k=100 \
#     --batch_size=512 > experiments/base/LLM_retrieval/log/train_val_nohup_rq.out 2>&1 &


# # topiocqa
# trained_model_path="3ricL/ad-hoc-ance-msmarco"
# # trained_model_path="experiments/base/-best-model"
# test_file_path="data/topiocqa/test_with_rewrite.json"
# new_query_jsonl="data/topiocqa/LLM_retrieval/generated_queries_rq_num5.jsonl"
# # new_query_jsonl="data/topiocqa/LLM_retrieval/generated_queries_rq.jsonl"
# output_trec_file="experiments/base_topiocqa/LLM_retrieval/test/qrecc_trec_llm_rq_num5_200"
# trec_gold_qrel_file_path="data/topiocqa/topiocqa_qrel.tsv"
# output_high_topk_path="experiments/base_topiocqa/LLM_retrieval/test/high_topk_rq_num5_200.json"

# # CUDA_VISIBLE_DEVICES=0 
# CUDA_VISIBLE_DEVICES=4  nohup python src/LLM_retrieval/test_query2query.py --pretrained_encoder_path=$trained_model_path \
#     --test_file_path=$test_file_path \
#     --new_query_jsonl=$new_query_jsonl \
#     --output_trec_file=$output_trec_file \
#     --trec_gold_qrel_file_path=$trec_gold_qrel_file_path \
#     --output_high_topk_path=$output_high_topk_path \
#     --max_concat_length=512 \
#     --top_k=200 \
#     --query_count=5 \
#     --dataset="topiocqa" \
#     --batch_size=512 > experiments/base_topiocqa/LLM_retrieval/log/test_nohup_rq_num5_200.out 2>&1 &


# # topiocqa  testquery rewrite alignment
# trained_model_path="experiments/base_acne_rewrite/output/0-model-epoch5-step8526-loss0.0055"
# # trained_model_path="experiments/base/-best-model"
# test_file_path="data/topiocqa/test_with_rewrite.json"
# new_query_jsonl="data/topiocqa/LLM_retrieval/generated_queries_rq_num5.jsonl"
# # new_query_jsonl="data/topiocqa/LLM_retrieval/generated_queries_rq.jsonl"
# output_trec_file="experiments/base_acne_rewrite/test/qrecc_trec_llm_rq_num5_100_5"
# trec_gold_qrel_file_path="data/topiocqa/topiocqa_qrel.tsv"
# output_high_topk_path="experiments/base_acne_rewrite/test/high_topk_rq_num5_100_5.json"

# # CUDA_VISIBLE_DEVICES=0 
# CUDA_VISIBLE_DEVICES=2  nohup python src/LLM_retrieval/test_query2query.py --pretrained_encoder_path=$trained_model_path \
#     --test_file_path=$test_file_path \
#     --new_query_jsonl=$new_query_jsonl \
#     --output_trec_file=$output_trec_file \
#     --trec_gold_qrel_file_path=$trec_gold_qrel_file_path \
#     --output_high_topk_path=$output_high_topk_path \
#     --max_concat_length=512 \
#     --top_k=100 \
#     --query_count=5 \
#     --dataset="topiocqa" \
#     --batch_size=512 > experiments/base_acne_rewrite/log/test_nohup_rq_num5_100_5.out 2>&1 &


# cast19
trained_model_path="3ricL/ad-hoc-ance-msmarco"
# trained_model_path="experiments/base/-best-model"
test_file_path="data/cast/data_cast19/new_data/cast19_test_topiocqa_with_pos_pids.jsonl"
new_query_jsonl="data/cast/data_cast19/new_data/cast19_test_passages_num5.jsonl"
output_trec_file="experiments/cast/cast19/test/cast_ance_base_num5"
trec_gold_qrel_file_path="data/cast/data_cast19/new_data/cast19_qrel.tsv"
output_high_topk_path="experiments/base/LLM_retrieval/test/cast19_ance_base_num5.json"

CUDA_VISIBLE_DEVICES=0  nohup python src/retrieval/test_conv2query.py --pretrained_encoder_path=$trained_model_path \
    --test_file_path=$test_file_path \
    --new_query_jsonl=$new_query_jsonl \
    --output_trec_file=$output_trec_file \
    --trec_gold_qrel_file_path=$trec_gold_qrel_file_path \
    --output_high_topk_path=$output_high_topk_path \
    --max_concat_length=512 \
    --top_k=100 \
    --batch_size=512 > experiments/cast/cast19/log/cast19_ance_base_num5.out 2>&1 &
