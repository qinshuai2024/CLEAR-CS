"""Test effectiveness of reranking model"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
import argparse
from utils.utils import  set_seed
from dataset.data import TestRerankQreccDataset
from os.path import join as oj
import json
from torch.utils.data import DataLoader
import torch
import numpy as np
import pytrec_eval
from transformers import AutoTokenizer
from model.reranker import QANLI4WayReranker,QANLIReranker
from tqdm import tqdm
from transformers import RobertaConfig, RobertaTokenizer

def output_test_res(query_embedding2id,
                    retrieved_scores_mat, 
                    retrieved_pid_mat,
                    offset2pid,
                    output_high_topk_path, # Top-10 ranked passages
                    args, top_k):   #  topk=10, 
    

    qids_to_ranked_candidate_passages = {}    # Mapping from sample_id to (pred_pid, score)
    # topN = args.top_k
    topN = top_k

    for query_idx in range(len(retrieved_pid_mat)):
        seen_pid = set()
        query_id = query_embedding2id[query_idx]  # sample_id

        top_ann_pid = retrieved_pid_mat[query_idx].copy()
        top_ann_score = retrieved_scores_mat[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        selected_ann_score = top_ann_score[:topN]
        rank = 0
        # if len(selected_ann_idx) < 199:
        #     print(f"Insufficient input selected_ann_idx:{len(selected_ann_idx)}")
        if query_id in qids_to_ranked_candidate_passages:
            pass
        else:
            tmp = [(0, 0)] * topN
            qids_to_ranked_candidate_passages[query_id] = tmp

        for idx, score in zip(selected_ann_idx, selected_ann_score):
            pred_pid = idx  # Direct passage number, no need to remap
            # No deduplication needed here
            qids_to_ranked_candidate_passages[query_id][rank] = (pred_pid, score)
            rank += 1
            # if not pred_pid in seen_pid: 
            #     qids_to_ranked_candidate_passages[query_id][rank] = (pred_pid, score)
            #     rank += 1
            #     seen_pid.add(pred_pid)


    # for case study and more intuitive observation
    logger.info('Loading query and passages\' real text...')
    
    # Save top-10 IDs   {query_id : [pred_pid]}
    # Preprocessing
    qid2topk = {
        qid: [pid_score[0] for pid_score in pid_scores[:top_k]]  # Take top-k pids
        for qid, pid_scores in qids_to_ranked_candidate_passages.items()  
    }
    with open(output_high_topk_path, "w") as f:
        json.dump(qid2topk, f, indent=4, ensure_ascii=False)
    logger.info(f"Successfully saved top-{top_k} relevant passages for training data")
    
    # qid2query = {}
    # with open(args.test_file_path, 'r') as f:
    #     data = f.readlines()
    # for record in data:
    #     record = json.loads(record.strip())
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
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.1", "recall.5", "recall.10", "recall.100", "recall.200"})   # 
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]
    recall_1_list = [v['recall_1'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_200_list = [v['recall_200'] for v in res.values()]

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
            "Recall@100": round(np.average(recall_100_list)*100, 5),
            "Recall@200": round(np.average(recall_200_list)*100, 5),
        }

    
    logger.info("---------------------Evaluation results:---------------------")   
    logger.info(res) 
    return res


def main():
    args = get_args()
    set_seed(args) 

    # 1. Load model
    tokenizer = AutoTokenizer.from_pretrained(args.reranker_encoder_path)
    
    # Use ANCE
    # model = Reranker(args.reranker_encoder_path).to(args.device)
    
    # # Use extractive QA (old implementation)
    # model = QAReranker(model_name=args.reranker_encoder_path)          # Ranking only
    # # model = QANLIReranker(model_name=args.reranker_encoder_path)     # With NLI distillation
    # state_dict = torch.load(args.save_path, map_location=args.device)
    # # Handle DDP-saved "module." prefix
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     if k.startswith("module."):
    #         new_state_dict[k[len("module."):]] = v
    #     else:
    #         new_state_dict[k] = v
    # model.load_state_dict(new_state_dict)
    # model = model.eval().to(args.device)

    # New 4-way classification reranking model + NLI distillation
    # model = QANLI4WayReranker(model_name=args.reranker_encoder_path)
    model = QANLIReranker(model_name=args.reranker_encoder_path)
    # model = QANLI4WayRerankerFromANCE(model_name=args.reranker_encoder_path)  # Using ANCE performs poorly
    state_dict = torch.load(args.save_path, map_location="cpu")
    # Handle DDP-saved "module." prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    model = model.eval().to(args.device)

    # Torch 2.0 acceleration
    try:
        model = torch.compile(model)
    except Exception as e:
        logger.warning(f"torch.compile not available: {e}")

    # 2. Load data
    dataset = TestRerankQreccDataset(
        file_path=args.test_file_path,
        tokenizer=tokenizer,
        rank_k=args.rank_k,
        dataset=args.dataset
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_gpu_test_batch_size,
        collate_fn=TestRerankQreccDataset.get_collate_fn(pad_token_id=pad_id, global_max_length=args.max_concat_length),
        num_workers=4,
        pin_memory=True
    )

    # 3. Inference and reranking
    retrieved_scores_mat = []
    retrieved_pid_mat = []
    query_embedding2id = []
    offset2pid = []

    logger.info("Starting inference...")
    with torch.inference_mode(), torch.cuda.amp.autocast():
        for batch in tqdm(dataloader, disable=args.disable_tqdm): 
            input_ids = batch["combined"].to(args.device, non_blocking=True)        # [B, 10, L]
            attention_mask = batch["combined_mask"].to(args.device, non_blocking=True)  # [B, 10, L]
            doc_ids_batch = batch["doc_ids"]   # list of [10] docid
            sample_ids = batch["sample_ids"]   # list of sample_id
            # Only use similarity scores
            # pos_probs, entail_probs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # New model output: classification logits + entailment probabilities
            class_logits, entail_probs = model(input_ids=input_ids, attention_mask=attention_mask)  # [B, K, 4], [B, K]
            args.tau_rel = 1
            pos_probs = torch.sigmoid(class_logits / args.tau_rel)  
            if args.mode == "all":
                relevance_scores = pos_probs + args.alpha_nli * entail_probs
            elif args.mode == "cls":
                relevance_scores = pos_probs  # Test classification cross-entropy
            elif args.mode == "nli":
                relevance_scores = entail_probs  # Test entailment distillation
            
            for scores, doc_ids, sample_id in zip(relevance_scores, doc_ids_batch, sample_ids):
                scores = scores.squeeze(0) if scores.dim() > 1 else scores  # [K]   Squeeze dimension 0
                # First get top-k
                top_scores, top_indices = torch.topk(scores, k=min(args.rank_k, scores.size(0)), dim=-1)
            
                seen = set()
                unique_pids, unique_scores = [], []
                for idx, score in zip(top_indices.cpu().tolist(), top_scores.cpu().tolist()):
                    if len(unique_pids) >= args.top_k:
                        break
                    pid = doc_ids[idx]
                    if pid not in seen:   # Deduplication
                        seen.add(pid)
                        unique_pids.append(pid)
                        unique_scores.append(score)
                    # Skip if already seen
                # logger.info(f"len(unique_pids): {len(unique_pids)}")
                # logger.info(f"len(unique_scores): {len(unique_scores)}")
                retrieved_scores_mat.append(unique_scores)
                retrieved_pid_mat.append(unique_pids)
                query_embedding2id.append(sample_id)
                offset2pid.extend(doc_ids)

    logger.info(f"retrieved_scores_mat: {len(retrieved_scores_mat)} queries")
    logger.info(f"retrieved_pid_mat[0]: {len(retrieved_pid_mat[0])} passages")

    # 4. Calculate metrics
    trec_result = output_test_res(
        query_embedding2id=query_embedding2id,
        retrieved_scores_mat=retrieved_scores_mat,
        retrieved_pid_mat=retrieved_pid_mat,
        offset2pid={i: pid for i, pid in enumerate(offset2pid)},
        output_high_topk_path=args.output_high_topk_path,
        args=args,
        top_k=args.top_k
    )
    # logger.info(trec_result)
    logger.info("Test finish!")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--reranker_encoder_path", type=str)
    parser.add_argument("--save_path", type=str)
    
    parser.add_argument("--test_file_path", type=str, default="datasets/topiocqa/test.json")
    parser.add_argument("--passage_collection_path", type=str, default="datasets/topiocqa/full_wiki_segments.tsv")
    parser.add_argument("--passage_embeddings_dir_path", type=str, default="datasets/topiocqa/embeds")
    parser.add_argument("--passage_offset2pid_path", type=str, default="datasets/topiocqa/tokenized/offset2pid.pickle")
    # parser.add_argument("--pretrained_encoder_path", type=str)
    parser.add_argument("--qrel_output_path", type=str, default="output/topiocqa")
    parser.add_argument("--output_trec_file", type=str)
    parser.add_argument("--trec_gold_qrel_file_path", type=str, default="datasets/topiocqa/topiocqa_qrel.trec")
    parser.add_argument("--dataset", type=str, default="topiocqa")
    parser.add_argument("--collate_fn_type", type=str, default="flat_concat_for_test")
    parser.add_argument("--output_high_topk_path", type=str, default="")

    parser.add_argument("--test_type", type=str, default="convqa")
    parser.add_argument("--is_train", type=bool, default=False)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--rank_k", type=int, default=100)
    
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
    parser.add_argument("--alpha_nli", type=float, default=1.0)
    parser.add_argument("--mode", type=str, required=True)  # cls  nli all  

    args = parser.parse_args()
    if args.use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    args.device = device

    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    return args

if __name__ == "__main__":
    main()
    

    