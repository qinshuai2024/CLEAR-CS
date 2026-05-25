# from IPython import embed
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
import csv
import argparse
from model.ance import ANCE
from utils.utils import set_seed
from dataset.data import padding_seq_to_same_length
from dataset.topiocqa import TopiocqaDataset
from dataset.qrecc import QReccDataset
import os
from os.path import join as oj
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import faiss
import time
import copy
import pickle
import torch
import numpy as np
import pytrec_eval
from transformers import RobertaConfig, RobertaTokenizer


def build_faiss_index(args):
    logger.info("Building index...")
    # ngpu = faiss.get_num_gpus()
    ngpu = args.n_gpu
    gpu_resources = []
    tempmem = -1

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    cpu_index = faiss.IndexFlatIP(768)  
    index = None
    if args.use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        # gpu_vector_resources, gpu_devices_vector
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres,
                                                    vdev,
                                                    cpu_index, co)
        index = gpu_index
    else:
        index = cpu_index

    return index


def search_one_by_one_with_faiss(args, passge_embeddings_dir, index, query_embeddings, topN):
    """
    Retrieve chunk by chunk, matching all query_emb with all passage_emb in current chunk
    """
    merged_candidate_matrix = None

    for block_id in range(args.passage_block_num):
        logger.info("Loading passage block " + str(block_id))
        passage_embedding = None
        passage_embedding2id = None
        try:
            with open(
                    oj(
                        passge_embeddings_dir,
                        "passage_emb_block_" + str(block_id) + ".pb"),
                    'rb') as handle:
                passage_embedding = pickle.load(handle)
            with open(
                    oj(
                        passge_embeddings_dir,
                        "passage_embid_block_" + str(block_id) + ".pb"),
                    'rb') as handle:
                passage_embedding2id = pickle.load(handle)
        except:
            break
        logger.info('passage embedding shape: ' + str(passage_embedding.shape))
        logger.info("query embedding shape: " + str(query_embeddings.shape))
        index.add(passage_embedding)

        # ann search
        tb = time.time()
        D, I = index.search(query_embeddings, topN) # D is score array(n_query, topN), I is passage index in faiss
        elapse = time.time() - tb
        logger.info({
            'time cost': elapse,
            'query num': query_embeddings.shape[0],
            'time cost per query': elapse / query_embeddings.shape[0]
        })
        # Convert to corresponding passage IDs
        candidate_id_matrix = passage_embedding2id[I] # passage_idx -> passage_id
        D = D.tolist()
        candidate_id_matrix = candidate_id_matrix.tolist()
        candidate_matrix = []   # n_query * top_N * 2   (score, passage_id)
        # Iterate through top-scoring passages for each query, save (score, passage_id)
        for score_list, passage_list in zip(D, candidate_id_matrix):
            candidate_matrix.append([])
            for score, passage in zip(score_list, passage_list):
                candidate_matrix[-1].append((score, passage))
            assert len(candidate_matrix[-1]) == len(passage_list)
        assert len(candidate_matrix) == I.shape[0]

        index.reset()
        del passage_embedding
        del passage_embedding2id

        if merged_candidate_matrix == None:
            merged_candidate_matrix = candidate_matrix
            continue
        
        # merge - combine scored passages for each query
        merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
        merged_candidate_matrix = []
        for merged_list, cur_list in zip(merged_candidate_matrix_tmp,
                                         candidate_matrix):
            p1, p2 = 0, 0
            merged_candidate_matrix.append([])
            while p1 < topN and p2 < topN:
                if merged_list[p1][0] >= cur_list[p2][0]:  # 0是得分
                    merged_candidate_matrix[-1].append(merged_list[p1])
                    p1 += 1
                else:
                    merged_candidate_matrix[-1].append(cur_list[p2])
                    p2 += 1
            while p1 < topN:
                merged_candidate_matrix[-1].append(merged_list[p1])
                p1 += 1
            while p2 < topN:
                merged_candidate_matrix[-1].append(cur_list[p2])
                p2 += 1

    merged_D, merged_I = [], []
    for merged_list in merged_candidate_matrix: # len(merged_candidate_matrix) = query_nums len([0]) = query_num * topk
        merged_D.append([])
        merged_I.append([])
        for candidate in merged_list: # len(merged_list) = query_num * topk
            merged_D[-1].append(candidate[0])
            merged_I[-1].append(candidate[1])
    
    merged_D, merged_I = np.array(merged_D), np.array(merged_I) # n_query * Top_N

    # logger.info(merged_I)
    logger.info(merged_I.shape)
    return merged_D, merged_I


def load_enhanced_model(model_path, original_encoder):
    """Load EnhancedQueryEncoder (not available by default)."""
    try:
        from model.enhanced import EnhancedQueryEncoder  # type: ignore
    except Exception as e:
        raise ImportError("EnhancedQueryEncoder is not available in this project") from e
    # 
    with open(os.path.join(model_path, 'enhanced_config.json')) as f:
        enhanced_config = json.load(f)
    model = EnhancedQueryEncoder(
        original_encoder=original_encoder,
        hidden_dim=enhanced_config['hidden_dim']
    )
    state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'))
    model.enhance_net.load_state_dict(state_dict['enhanced_state_dict'])
    return model

def get_test_query_embedding(args):
    set_seed(args)

    config = RobertaConfig.from_pretrained(args.pretrained_encoder_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_encoder_path, do_lower_case=True)
    model = ANCE.from_pretrained(args.pretrained_encoder_path, config=config).to(args.device)
    
    # test dataset/dataloader
    args.batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)
    logger.info("Buidling test dataset...")
    if args.dataset == "topiocqa":
        test_dataset = TopiocqaDataset(args, tokenizer, args.test_file_path)
    elif args.dataset == "qrecc":
        test_dataset = QReccDataset(args, tokenizer, args.test_file_path)
    test_loader = DataLoader(test_dataset, 
                                batch_size = args.batch_size, 
                                shuffle=False, 
                                collate_fn=test_dataset.get_collate_fn(args))
    

    logger.info("Generating query embeddings for testing...")
    model.zero_grad()
    # hard_expert.zero_grad()

    embeddings = []
    embedding2id = []

    with torch.no_grad():
        for batch in tqdm(test_loader, disable=args.disable_tqdm):
            model.eval()
            bt_sample_ids = batch["bt_sample_ids"] # question id
            if args.dataset in ["topiocqa", "qrecc"]:
                input_ids = batch["bt_conv_qa"].to(args.device)
                input_masks = batch["bt_conv_qa_mask"].to(args.device)
                # input_ids = batch["bt_oracle_utt"].to(args.device)
                # input_masks = batch["bt_oracle_utt_mask"].to(args.device)
            else:
                input_ids = batch["bt_input_ids"].to(args.device)
                input_masks = batch["bt_attention_mask"].to(args.device)
                # input_ids = batch["bt_oracle_labels"].to(args.device)
                # input_masks = batch["bt_oracle_labels_mask"].to(args.device)

            query_embs = model(input_ids, input_masks)
            # query_embs = hard_expert(outputs)
            query_embs = query_embs.detach().cpu().numpy()
            embeddings.append(query_embs)
            embedding2id.extend(bt_sample_ids)

    embeddings = np.concatenate(embeddings, axis = 0)
    torch.cuda.empty_cache()

    return embeddings, embedding2id


def output_test_res(query_embedding2id,
                    retrieved_scores_mat, # score_mat: score matrix, test_query_num * (top_k * block_num)
                    retrieved_pid_mat, # pid_mat: corresponding passage ids
                    offset2pid,
                    output_high_topk_path, # 排名前10的段落
                    args, topk=100):
    

    qids_to_ranked_candidate_passages = {}    # sample_id 与 (pred_pid, score)的映射
    topN = args.top_k

    for query_idx in range(len(retrieved_pid_mat)):
        seen_pid = set()
        query_id = query_embedding2id[query_idx]  # sample_id

        top_ann_pid = retrieved_pid_mat[query_idx].copy()
        top_ann_score = retrieved_scores_mat[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        selected_ann_score = top_ann_score[:topN].tolist()
        rank = 0

        if query_id in qids_to_ranked_candidate_passages:
            pass
        else:
            tmp = [(0, 0)] * topN
            tmp_ori = [0] * topN
            qids_to_ranked_candidate_passages[query_id] = tmp

        for idx, score in zip(selected_ann_idx, selected_ann_score):
            pred_pid = offset2pid[idx]  # 下标就是索引，映射到p_id

            if not pred_pid in seen_pid: 
                qids_to_ranked_candidate_passages[query_id][rank] = (pred_pid, score)
                rank += 1
                seen_pid.add(pred_pid)


    # for case study and more intuitive observation
    logger.info('Loading query and passages\' real text...')
    
    # # 0517 保存前10个id   {query_id : [pred_pid]}
    # # 预处理
    qid2topk = {
        qid: [pid_score[0] for pid_score in pid_scores[:topk]]  #  取前10个pid
        for qid, pid_scores in qids_to_ranked_candidate_passages.items()  
    }
    with open(output_high_topk_path, "w") as f:
        json.dump(qid2topk, f, indent=4, ensure_ascii=False)
    logger.info(f"成功保存数据的前{topk}相关段落")
    
    qid2query = {}
    with open(args.test_file_path, 'r') as f:
        data = f.readlines()
    for record in data:
        record = json.loads(record.strip())
        #qid2query[record["sample_id"]] = record["query"]
    
    

    # write to file
    logger.info('begin to write the output...')

    output_trec_file = oj(args.qrel_output_path, args.output_trec_file)
    with open(output_trec_file, "w") as g:
        for qid, passages in qids_to_ranked_candidate_passages.items():
            #query = qid2query[qid]
            rank_list = []
            for i in range(topN):
                pid, score = passages[i]
                g.write(str(qid) + " Q0 " + str(pid) + " " + str(i + 1) + " " + str(-i - 1 + 200) + ' ' + str(score) + " ance\n")

    logger.info("output file write ok at {}".format(output_trec_file))
    trec_res = print_trec_res(output_trec_file, args.trec_gold_qrel_file_path, args.rel_threshold)
    return trec_res

def print_trec_res(run_file, qrel_file, rel_threshold=1):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    
    for line in qrel_data:
        # line = line.split(" ")
        line = line.strip().split("\t")
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(line[4])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.1", "recall.5", "recall.10", "recall.20", "recall.100"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]
    recall_1_list = [v['recall_1'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    res = {
            "MAP": round(np.average(map_list)*100, 5),
            "MRR": round(np.average(mrr_list)*100, 5),
            "NDCG@3": round(np.average(ndcg_3_list)*100, 5), 
            "Recall@1": round(np.average(recall_1_list)*100, 5),
            "Recall@5": round(np.average(recall_5_list)*100, 5),
            "Recall@10": round(np.average(recall_10_list)*100, 5),
            #"Recall@20": round(np.average(recall_20_list)*100, 5),
            "Recall@100": round(np.average(recall_100_list)*100, 5),
        }

    
    logger.info("---------------------Evaluation results:---------------------")    
    logger.info(res)
    return res

def gen_metric_score_and_save(args, index, query_embeddings, query_embedding2id, output_high_topk_path):
    # score_mat: score matrix, test_query_num * (top_n * block_num)
    # pid_mat: corresponding passage ids
    retrieved_scores_mat, retrieved_pid_mat = search_one_by_one_with_faiss(
                                                     args,
                                                     args.passage_embeddings_dir_path, 
                                                     index, 
                                                     query_embeddings, 
                                                     args.top_k) 

    with open(args.passage_offset2pid_path, "rb") as f:
        offset2pid = pickle.load(f)
    
    output_test_res(query_embedding2id,
                    retrieved_scores_mat,
                    retrieved_pid_mat,
                    offset2pid,
                    output_high_topk_path,
                    args)


def main():
    args = get_args()
    set_seed(args) 
    
    index = build_faiss_index(args)
    query_embeddings, query_embedding2id = get_test_query_embedding(args)  # query_embedding2id 一个一维列表，存放下标对应query的sample_id
    gen_metric_score_and_save(args, index, query_embeddings, query_embedding2id, args.output_high_topk_path)

    logger.info("Test finish!")
    

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_file_path", type=str, default="datasets/topiocqa/test.json")
    parser.add_argument("--passage_collection_path", type=str, default="datasets/topiocqa/full_wiki_segments.tsv")
    parser.add_argument("--passage_embeddings_dir_path", type=str, default="datasets/topiocqa/embeds")
    parser.add_argument("--passage_offset2pid_path", type=str, default="datasets/topiocqa/tokenized/offset2pid.pickle")
    parser.add_argument("--pretrained_encoder_path", type=str)
    parser.add_argument("--qrel_output_path", type=str, default="output/topiocqa")
    parser.add_argument("--output_trec_file", type=str)
    parser.add_argument("--trec_gold_qrel_file_path", type=str, default="datasets/topiocqa/topiocqa_qrel.trec")
    parser.add_argument("--dataset", type=str, default="topiocqa")
    parser.add_argument("--collate_fn_type", type=str, default="flat_concat_for_test")  # flat_concat_for_train
    parser.add_argument("--output_high_topk_path", type=str, default="")

    parser.add_argument("--test_type", type=str, default="convqa")
    parser.add_argument("--is_train", type=bool, default=False)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--rel_threshold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_gpu_test_batch_size", type=int, default=4)
    parser.add_argument("--passage_block_num", type=int, default=26) # 22 for qrecc and 26 for topiocqa
    parser.add_argument("--disable_tqdm", type=bool, default=False)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--use_data_percent", type=float, default=1)
    
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_doc_length", type=int, default=384)
    parser.add_argument("--max_response_length", type=int, default=32)
    parser.add_argument("--max_concat_length", type=int, default=512)

    args = parser.parse_args()
    if args.use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    args.device = device

    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    return args


import debugpy

if __name__ == '__main__':
    
    # debugpy.listen(5678)  # 默认端口
    # print("等待调试器附加...")  # 可选提示
    # debugpy.wait_for_client()  # 阻塞直到调试器连接
    # print("Debugger attached, starting execution...")  # Optional prompt
    main()
    # Example usage:
    # run_file = "experiments/ance/test/topiocqa_trec_results.trec"
    # qrel_file = "data/topiocqa/qrel.trec"
    # print_trec_res(run_file, qrel_file)

