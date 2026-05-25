"""
combine_rerank_data.py

Compute HNAP (Hard Negatives Above Positives) metric @k = {# hard negatives ranked above positives in top k}/{total # hard negatives in top k}

Main steps:
1. Collect passages: Save top-10 passage IDs from test results
2. Combine data: Combine each query with corresponding 10 passage IDs and text
3. GPT identification: Use GPT with three labels - fully relevant with correct answer, relevant but no answer, completely irrelevant
4. Collect labels: Collect labels for each query and compute average metrics

Usage:
    - Input: Top-10 passage IDs from test results
    - Output: Average HNAP score
"""



import time
import json
import logging
import traceback
from collections import defaultdict

from tqdm import tqdm
from openai import OpenAI



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # , filename="experiments/base/test/hnap_metric.log"
logger = logging.getLogger(__name__)

def combine_query_with_top_passages(topk_id_path, test_path, qrecc_collection_path, query_topk_passages_path, dataset="qrecc"):
    """组合数据: 组合每个查询与对应的topk个段落的ID和文本"""
    # 1.读取tok_ids
    with open(topk_id_path, "r") as f:
        topk_ids = json.load(f)
    for k, v in topk_ids.items():   # 将llm的检索结果的pid转为int，因为llm的那个是字符串
        topk_ids[k] = [int(x) for x in v]
    
    # 2.设置需要的pid,并保存对应文本
    needed_pids_set = [pid for pids in topk_ids.values() for pid in pids]
    # print(len(needed_pids_set))
    needed_pids_set = set(needed_pids_set)
    # load collection.tsv
    pid2doc = {}
    if dataset == "topiocqa":
        num_num_doc = 25700592
    elif dataset == "qrecc":
        num_num_doc = 54573064
    elif dataset == "cast1920":
        num_num_doc = 35848188
    
    bad_doc_set = set()
    print(f"Loading QReCC collection, total {num_num_doc} passages...")
    for line in tqdm(open(qrecc_collection_path, "r"), total=num_num_doc):
        try:
            sample = line.strip().split('\t')
            if dataset=="topiocqa":
                pid, doc, title = sample
                if pid == "id":
                    continue
                title = " ".join(title.split(' [SEP] '))
                doc = " ".join([title, doc.rstrip()])
            else:
                pid, doc = sample
            pid = int(pid)
            if pid in needed_pids_set:   # 只保存需要的，且正常有文本的段落
                pid2doc[pid] = doc
        except:
            pid = int(line.strip().split('\t')[0])
            doc = ""
            bad_doc_set.add(pid)
    logger.info("Loadding QReCC collection OK! Total bad passages = {}".format(len(bad_doc_set)))
    
    # 3.pid转文本
    with open(test_path, "r") as f:
        test_data = f.readlines()
    new_data = []
    cnt = 0
    for record in test_data:
        record = json.loads(record.strip())
        # 只保留有正样本的，因为计算指标也只计算这些
        if len(record["pos_docs_pids"]) > 0:
            qid = record["sample_id"]
            topk_id = topk_ids[qid]
            record["topk_id"] = topk_id
            try:
                record["topk_text"] = [pid2doc[pid] for pid in topk_id]
            except Exception as e:
                print(qid)
                print(topk_id)
                return
            new_data.append(record)
            cnt += 1
    logger.info(f"段落id转文本成功, 共{cnt}个测试数据")
    
    # 4.保存新数据
    with open(query_topk_passages_path, "w") as f:
        for record in new_data:
            f.write(json.dumps(record) + '\n')
            
    logger.info("保存成功") 
    

def combine_query_with_top_passages_llm_base(topk_id_path, topk_id_path_llm ,test_path, qrecc_collection_path, \
    query_topk_passages_path, dataset='qrecc', ance_top_k=100, llm_top_k=100):
    """组合数据, 对conv2query和dense的进行组合: 组合每个查询与对应的topk个段落的ID和文本"""
    # 1.读取tok_ids
    with open(topk_id_path, "r") as f:
        topk_ids = json.load(f)
    with open(topk_id_path_llm, "r") as f:
        topk_llm_ids = json.load(f)
    for k, v in topk_llm_ids.items():   # 将llm的检索结果的pid转为int，因为llm的那个是字符串
        topk_llm_ids[k] = [int(x) for x in v]
    # 2.设置需要的pid,并保存对应文本
    needed_pids_set = [pid for pids in topk_ids.values() for pid in pids] + [pid for pids in topk_llm_ids.values() for pid in pids]
    needed_pids_set = set(needed_pids_set)
    # load collection.tsv
    pid2doc = {}
    if dataset == "topiocqa":
        num_num_doc = 25700592
    elif dataset == "qrecc":
        num_num_doc = 54573064
    elif dataset == "cast1920":
        num_num_doc = 35848188
    elif dataset == "cast21":
        num_num_doc = 45565388
    
    bad_doc_set = set()
    print(f"Loading QReCC collection, total {num_num_doc} passages...")
    for line in tqdm(open(qrecc_collection_path, "r"), total=num_num_doc):
        try:
            sample = line.strip().split('\t')
            if dataset=="topiocqa":
                pid, doc, title = sample
                if pid == "id":
                    continue
                title = " ".join(title.split(' [SEP] '))
                doc = " ".join([title, doc.rstrip()])
            else:
                pid, doc = sample
            pid = int(pid)
            if pid in needed_pids_set: 
                pid2doc[pid] = doc
        except:
            pid = int(line.strip().split('\t')[0])
            doc = ""
            bad_doc_set.add(pid)
    
    logger.info("Loadding QReCC collection OK! Total bad passages = {}".format(len(bad_doc_set)))
    
    # 3.pid转文本
    with open(test_path, "r") as f:
        test_data = f.readlines()
    new_data = []
    cnt = 0
    for record in test_data:
        record = json.loads(record.strip())
        # 只保留有正样本的，因为计算指标也只计算这些
        if len(record["pos_docs_pids"]) > 0:
            qid = record["sample_id"]
            topk_id = topk_ids[qid][:ance_top_k]
            topk_llm_id = topk_llm_ids[qid][:llm_top_k]
            all_pid = topk_id + topk_llm_id
            record["topk_id"] = all_pid
            record["topk_text"] = [pid2doc[pid] for pid in all_pid]
            new_data.append(record)
            cnt += 1
    logger.info(f"段落id转文本成功, 共{cnt}个测试数据")
    
    # 4.保存新数据
    with open(query_topk_passages_path, "w") as f:
        for record in new_data:
            f.write(json.dumps(record) + '\n')
            
    logger.info("保存成功")    


if __name__ == "__main__":
    # topk_id_path = "experiments/base/LLM_retrieval/test/high_topk_rq_all.json"
    # test_path = "data/qrecc/new_preprocessed/test.json"
    # qrecc_collection_path = "data/qrecc/new_preprocessed/qrecc_collection.tsv"
    # query_topk_passages_path = "experiments/base/LLM_retrieval/test/high_topk_text_no_pos_llm_rq_all.json"
    # combine_query_with_top_passages_nopos(topk_id_path, test_path, qrecc_collection_path, query_topk_passages_path)
    # # combine_query_with_top_passages(topk_id_path, test_path, qrecc_collection_path, query_topk_passages_path)
    
    # # 0920  保存训练数据查询和前100段落文本
    # topk_id_path = "experiments/base/base_train/high_top_100.json"
    # train_path = "data/qrecc/new_preprocessed/train.json"
    # qrecc_collection_path = "data/qrecc/new_preprocessed/qrecc_collection.tsv"
    # query_topk_passages_path = "experiments/base/base_train/train_high_top_100_text.json"
    # combine_query_with_top_passages(topk_id_path, train_path, qrecc_collection_path, query_topk_passages_path)
    
    # 测试数据topk id-text
    # topk_id_path = "experiments/base_ance/qrecc/data/test_high_top_100_ance.json"
    # test_path = "data/qrecc/new_preprocessed/test.json"
    # qrecc_collection_path = "data/qrecc/new_preprocessed/qrecc_collection.tsv"
    # query_topk_passages_path = "experiments/base_ance/qrecc/data/test_high_top_100_ance_text.json"
    # combine_query_with_top_passages(topk_id_path, test_path, qrecc_collection_path, query_topk_passages_path)
    
    
    # # topiocqa
    # topk_id_path = "data/topiocqa/recall/test_top_100_ance.json"  # 
    # test_path = "data/topiocqa/test.json"
    # query_topk_passages_path = "data/topiocqa/recall/test_top_100_ance_text.json"
    # # topk_id_path = "experiments/base_ance_cls/data/test_high_top_100_base_17.json"  # bm25 cls
    # # topk_id_path = "experiments/base_topiocqa/LLM_retrieval/test/high_topk_rq_num5_200.json"
    # # test_path = "data/topiocqa/test_with_rewrite.json"
    # collection_path = "data/topiocqa/full_wiki_segments.tsv"
    # # query_topk_passages_path = "experiments/base_ance/topiocqa/data/test_high_top_100_ance_bm25_text.json"
    # combine_query_with_top_passages(topk_id_path, test_path, collection_path, query_topk_passages_path, dataset="topiocqa")
    
    # # qrecc
    # # topk_id_path = "experiments/reranker/data/test_high_top_100.json"
    # topk_id_path = "experiments/base_ance/qrecc/data/test_high_top_100_ance.json"
    # # topk_id_llm_path = "experiments/base/LLM_retrieval/test/high_topk_rq_1.json"
    # topk_id_llm_path = "experiments/qrecc_ance/data/llm_topk_num10.json"
    # test_path = "data/qrecc/new_preprocessed/test.json"
    # qrecc_collection_path = "data/qrecc/new_preprocessed/qrecc_collection.tsv"
    # query_topk_passages_path = "data/qrecc/new_data/recall/test_topk_ance_llm_10.json"
    # combine_query_with_top_passages_llm_base(topk_id_path, topk_id_llm_path, test_path, qrecc_collection_path, \
    #     query_topk_passages_path, ance_top_k=100, llm_top_k=100)
    
    # # topiocqa
    # # topk_id_path = "experiments/base_topiocqa/data/test_high_top_100_base_13.json"  # qracdr
    # # topk_id_path = "experiments/base_ance_cls/data/test_high_top_100_base_17.json"  # bm25 cls
    # topk_id_path = "extracted_topk_ids.json"
    # # topk_id_llm_path = "experiments/base_topiocqa/LLM_retrieval/test/high_topk_rq_all.json"
    # # topk_id_llm_path = "experiments/base_topiocqa/LLM_retrieval/test/high_topk_rq_num3.json"
    # # topk_id_llm_path = "experiments/base_topiocqa/LLM_retrieval/test/high_topk_rq_num5_200.json"
    # # topk_id_llm_path = "experiments/base_acne_rewrite/test/high_topk_rq_num5_100_7.json"
    # topk_id_llm_path="experiments/topi_ance/data/llm_topk_num10.json"
    # test_path = "data/topiocqa/test_with_rewrite.json"
    # qrecc_collection_path = "data/topiocqa/full_wiki_segments.tsv"
    # query_topk_passages_path = "data/topiocqa/recall/test_ance_llm_num10.json"
    # combine_query_with_top_passages_llm_base(topk_id_path, topk_id_llm_path, test_path, qrecc_collection_path, \
    #     query_topk_passages_path, ance_top_k=100, llm_top_k=100, dataset='topiocqa')
    
    # # cast19 topk id-text
    # topk_id_path = "experiments/cast/cast19/test_qracdr.json"
    # test_path = "data/cast/data_cast19/new_data/cast19_test_topiocqa_with_pos_pids.jsonl"
    # qrecc_collection_path = "data/cast/cast2019.tsv"
    # query_topk_passages_path = "data/cast/data_cast19/new_data/test_high_top_100_qrecc_text.json"
    # combine_query_with_top_passages(topk_id_path, test_path, qrecc_collection_path, query_topk_passages_path, dataset="cast1920")
    
    
    # # topk_id_llm_path = "experiments/cast/cast19/cast19_topiocqa_llm_num5.json"
    # # topk_id_path = "experiments/cast/cast19/cast19_topiocqa_ance.json"
    # topk_id_llm_path = "experiments/cast/cast19/cast19_qrecc_llm_num5.json"
    # topk_id_path = "experiments/cast/cast19/cast19_qrecc_ance.json"
    # test_path = "data/cast/data_cast19/new_data/cast19_test_topiocqa_with_pos_pids.jsonl"
    # qrecc_collection_path = "data/cast/cast2019.tsv"
    # query_topk_passages_path = "data/cast/data_cast19/new_data/test_qrecc_llm5_text.json"
    # combine_query_with_top_passages_llm_base(topk_id_path, topk_id_llm_path, test_path, qrecc_collection_path, \
    #     query_topk_passages_path, ance_top_k=100, llm_top_k=100, dataset='cast1920')
    
    
    # cast20 topk id-text
    # topk_id_path = "experiments/cast/cast20/test_qracdr.json"
    # test_path = "data/cast/data_cast20/new_data/cast20_test_topiocqa_with_pos_pids.jsonl"
    # qrecc_collection_path = "data/cast/cast2019.tsv"
    # query_topk_passages_path = "data/cast/data_cast20/new_data/test_high_top_100_qrecc_text.json"
    # combine_query_with_top_passages(topk_id_path, test_path, qrecc_collection_path, query_topk_passages_path, dataset="cast1920")
    
    
    
    # # topk_id_llm_path = "data/cast/data_cast20/new_data/cast20_topiocqa_llm_num5.json"
    # # topk_id_path = "experiments/cast/cast20/cast20_topiocqa_ance.json"
    # topk_id_llm_path = "data/cast/data_cast20/new_data/cast20_qrecc_llm_num5.json"
    # topk_id_path = "experiments/cast/cast20/cast20_qrecc_ance.json"
    # test_path = "data/cast/data_cast20/new_data/cast20_test_topiocqa_with_pos_pids.jsonl"
    # qrecc_collection_path = "data/cast/cast2019.tsv"
    # query_topk_passages_path = "data/cast/data_cast20/new_data/test_qrecc_llm5_text.json"
    # combine_query_with_top_passages_llm_base(topk_id_path, topk_id_llm_path, test_path, qrecc_collection_path, \
    #     query_topk_passages_path, ance_top_k=100, llm_top_k=100, dataset='cast1920')
    
    # cast21
    topk_id_llm_path = "experiments/cast/cast21/cast21_qrecc_llm_num5.json"
    topk_id_path = "experiments/cast/cast21/cast21_qrecc_ance.json"
    test_path = "data/cast/data_cast21/new_data/cast21_test_topiocqa_with_pos_pids.jsonl"
    qrecc_collection_path = "data/cast/data_cast21/ori_data/passages.tsv"
    query_topk_passages_path = "data/cast/data_cast21/new_data/test_qrecc_llm5_text.json"
    combine_query_with_top_passages_llm_base(topk_id_path, topk_id_llm_path, test_path, qrecc_collection_path, \
        query_topk_passages_path, ance_top_k=100, llm_top_k=100, dataset='cast21')
