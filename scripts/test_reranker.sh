#! /bin/bash

# # trained_model_path="experiments/reranker/output/best_reranker_model.pt"     # 
# test_file_path="experiments/reranker/data/test_high_topk_text_llm_base_10_unique.json"
# reranker_encoder_path="deepset/roberta-base-squad2"  
# qrel_output_path="experiments/reranker/test"
# output_trec_file="qrecc_trec_best_top10_llm_base_reranker"
# trec_gold_qrel_file_path="data/qrecc/new_preprocessed/qrecc_qrel.tsv"
# output_high_topk_path=""
# save_path="experiments/reranker/output/best_reranker_model.pt"

# CUDA_VISIBLE_DEVICES=1 nohup python src/test_rerank.py \
#     --test_file_path=$test_file_path \
#     --reranker_encoder_path=$reranker_encoder_path \
#     --save_path=$save_path \
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
#     --top_k=20 \
#     --per_gpu_test_batch_size=8 > experiments/reranker/log/test_nohup_top10_llm_base_reranker_unique.out 2>&1 &


# # test_file_path="experiments/reranker/data/test_high_topk_text_llm_base_10.json"
# # test_file_path="experiments/reranker/data/test_high_topk_text.json"
# # test_file_path="experiments/base/base_train/train_high_top_100_text.json"
# # test_file_path="experiments/base_ance/qrecc/data/test_high_top_100_ance_text.json"
test_file_path="experiments/base_ance/qrecc/data/test_high_topk_text_llm_ance_100.json"
reranker_encoder_path="deepset/roberta-base-squad2"  
qrel_output_path="experiments/qrecc_reranker/test"
output_trec_file="qrecc_trec_best_top100_llm_ance_reranker_1"
trec_gold_qrel_file_path="data/qrecc/new_preprocessed/qrecc_qrel.tsv"
output_high_topk_path=""
save_path="experiments/qrecc_reranker/output/best_reranker_model_1.pt"

CUDA_VISIBLE_DEVICES=5 nohup python src/rerank/test_reranker.py \
    --test_file_path=$test_file_path \
    --reranker_encoder_path=$reranker_encoder_path \
    --save_path=$save_path \
    --qrel_output_path=$qrel_output_path \
    --output_trec_file=$output_trec_file \
    --trec_gold_qrel_file_path=$trec_gold_qrel_file_path \
    --output_high_topk_path=$output_high_topk_path \
    --test_type="convqa" \
    --max_query_length=32 \
    --max_doc_length=384 \
    --max_response_length=64 \
    --max_concat_length=512 \
    --dataset="qrecc" \
    --passage_block_num=22 \
    --top_k=100 \
    --rank_k=200 \
    --per_gpu_test_batch_size=8 > experiments/qrecc_reranker/log/test_nohup_top100_llm_ance_reranker_1.out 2>&1 &


# rank_kis the number of passages to rank
# top_kis the final number to keep



# # testNLI
# test_file_path="experiments/reranker/data/test_high_topk_text_llm_base_100.json"
# # test_file_path="experiments/reranker/data/test_high_topk_text.json"
# reranker_encoder_path="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# qrel_output_path="experiments/NLI/test"
# output_trec_file="qrecc_trec_best_top100_llm_base_NLI"
# trec_gold_qrel_file_path="data/qrecc/new_preprocessed/qrecc_qrel.tsv"
# output_high_topk_path=""
# ranker_save_path="experiments/reranker_1/output/best_reranker_model.pt"

# CUDA_VISIBLE_DEVICES=4 nohup python src/test_nli.py \
#     --test_file_path=$test_file_path \
#     --reranker_encoder_path=$reranker_encoder_path \
#     --save_path=$ranker_save_path \
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
#     --rank_k=200 \
#     --per_gpu_test_batch_size=4 > experiments/NLI/log/test_nohup_top100_llm_base_NLI.out 2>&1 &

# topiocqa
# test_file_path="experiments/base_ance/topiocqa/data/test_high_top_100_ance_text.json"
# test_file_path="experiments/base_ance/topiocqa/data/test_high_top_100_llm3_text.json"
# test_file_path="experiments/base_ance/topiocqa/data/test_high_topk_text_llm_ance_cls_100.json"
# test_file_path="experiments/base_ance/topiocqa/data/test_high_topk_text_llm_3_ance_cls_100.json"
# test_file_path="experiments/base_ance/topiocqa/data/test_high_topk_text_llm_5_qracdr.json"
# test_file_path="experiments/base_ance/topiocqa/data/test_high_topk_text_llm_5_ance_new.json"
# test_file_path="experiments/base_ance/topiocqa/data/test_high_topk_text_llm_5_conv.json"  # using LLM retrieval results from dialogue queries
# # test_file_path="experiments/base_ance/topiocqa/data/test_high_top_100_llm5_text.json"
# # test_file_path="experiments/base_ance/topiocqa/data/test_high_top_100_ance_bm25_text.json"
# # test_file_path="experiments/base_ance/topiocqa/data/train_high_top_100_qracdr_text.json"  # traindataset
# reranker_encoder_path="deepset/roberta-base-squad2"  
# # reranker_encoder_path="3ricL/ad-hoc-ance-msmarco" 
# qrel_output_path="experiments/topi_reranker_2waynocorr/test"
# output_trec_file="topiocqa_trec_best_top100_ance_cls_llm_reranker_4_new"    # ance_cls_   llm_
# trec_gold_qrel_file_path="data/topiocqa/topiocqa_qrel.tsv"
# # trec_gold_qrel_file_path="data/topiocqa/train_gold.tsv"
# output_high_topk_path="experiments/base_ance/topiocqa/data/train_test_high_top_100_qracdr.json"
# # save_path="experiments/reranker_topi_nli_4/output/best_reranker_model_5.pt" # current best reranker
# save_path="experiments/topi_reranker_2waynocorr/output/best_qanli_reranker_4.pt" # current best distillation reranker

# CUDA_VISIBLE_DEVICES=2 nohup python src/rerank/test_reranker.py \
#     --test_file_path=$test_file_path \
#     --reranker_encoder_path=$reranker_encoder_path \
#     --save_path=$save_path \
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
#     --rank_k=200 \
#     --alpha_nli=5 \
#     --per_gpu_test_batch_size=8 > experiments/topi_reranker_2waynocorr/log/test_nohup_top100_ance_cls_llm_4_5.out 2>&1 &

# # testNLI   topiocqa
# test_file_path="experiments/base_ance/topiocqa/data/test_high_topk_text_llm_5_ance_new.json"
# # test_file_path="experiments/reranker/data/test_high_topk_text.json"
# reranker_encoder_path="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# qrel_output_path="experiments/NLI/test"      
# output_trec_file="topiocqa_trec_best_top100_llm_base_NLI_query"    # using query+answer test，please modify as needed
# trec_gold_qrel_file_path="data/topiocqa/topiocqa_qrel.tsv"
# output_high_topk_path=""
# ranker_save_path="experiments/reranker_1/output/best_reranker_model.pt"

# CUDA_VISIBLE_DEVICES=1 nohup python src/test_nli.py \
#     --test_file_path=$test_file_path \
#     --reranker_encoder_path=$reranker_encoder_path \
#     --save_path=$ranker_save_path \
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
#     --passage_block_num=22 \
#     --top_k=100 \
#     --rank_k=200 \
#     --per_gpu_test_batch_size=4 > experiments/NLI/log/test_topiocqa_nohup_top100_llm5_ance_NLI_query.out 2>&1 &


# using ance
# test_file_path="experiments/base_ance/topiocqa/data/test_high_topk_text_llm_5_ance_new.json"
# # test_file_path="experiments/base_ance/topiocqa/data/test_high_top_100_llm5_text.json"
# # test_file_path="experiments/base_ance/topiocqa/data/test_high_top_100_ance_bm25_text.json"
# # reranker_encoder_path="deepset/roberta-base-squad2"  
# reranker_encoder_path="3ricL/ad-hoc-ance-msmarco" 
# qrel_output_path="experiments/reranker_tpoi_nli_ance/test"
# output_trec_file="topiocqa_trec_best_top100_llm_ance_cls_4"    # ance_cls_   llm_
# trec_gold_qrel_file_path="data/topiocqa/topiocqa_qrel.tsv"
# output_high_topk_path=""
# # save_path="experiments/reranker_topi_nli_4/output/best_reranker_model_5.pt" # current best reranker
# save_path="experiments/reranker_tpoi_nli_ance/output/best_reranker_model_4.pt" # current best distillation reranker

# CUDA_VISIBLE_DEVICES=0 nohup python src/test_re.py \
#     --test_file_path=$test_file_path \
#     --reranker_encoder_path=$reranker_encoder_path \
#     --save_path=$save_path \
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
#     --rank_k=200 \
#     --per_gpu_test_batch_size=8 > experiments/reranker_tpoi_nli_ance/log/test_nohup_top100_llm_ance_cls_4.out 2>&1 &




# # testcast19
# test_file_path="data/cast/data_cast19/new_data/test_high_top_100_qrecc_llm5_text.json" 
# reranker_encoder_path="deepset/roberta-base-squad2"  
# qrel_output_path="experiments/cast/cast19/reranker/test"
# output_trec_file="cast19_top100_qrecc_llm_reranker"    # qrecc_   llm_
# trec_gold_qrel_file_path="data/cast/data_cast19/new_data/cast19_qrel.tsv"
# output_high_topk_path=""
# save_path="experiments/topi_reranker_2waycorr/output/best_qanli_reranker_4.pt"

# CUDA_VISIBLE_DEVICES=6 nohup python src/rerank/test_reranker.py \
#     --test_file_path=$test_file_path \
#     --reranker_encoder_path=$reranker_encoder_path \
#     --save_path=$save_path \
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
#     --rank_k=100 \
#     --alpha_nli=1 \
#     --per_gpu_test_batch_size=8 > experiments/cast/cast19/reranker/log/test_top100_qrecc_llm5.out 2>&1 &


# # testcast20
# test_file_path="data/cast/data_cast20/new_data/test_high_top_100_qrecc_text.json" 
# reranker_encoder_path="deepset/roberta-base-squad2"  
# qrel_output_path="experiments/cast/cast20/reranker/test"
# output_trec_file="cast19_top100_qrecc_reranker"    # qrecc_   llm_
# trec_gold_qrel_file_path="data/cast/data_cast20/new_data/cast20_qrel.tsv"
# output_high_topk_path=""
# save_path="experiments/qrecc_reranker_nomlp/output/best_reranker_model_1.pt"

# CUDA_VISIBLE_DEVICES=0 nohup python src/rerank/test_reranker.py \
#     --test_file_path=$test_file_path \
#     --reranker_encoder_path=$reranker_encoder_path \
#     --save_path=$save_path \
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
#     --rank_k=100 \
#     --alpha_nli=1 \
#     --per_gpu_test_batch_size=8 > experiments/cast/cast20/reranker/log/test_top100_qrecc.out 2>&1 &

