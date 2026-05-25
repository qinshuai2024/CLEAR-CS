# coding: utf-8
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
# Get absolute path of current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get parent directory
parent_dir = os.path.dirname(current_dir)
# Add parent directory to Python path
sys.path.append(parent_dir)
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
import json
import logging
import argparse
import numpy as np
import faiss
import torch
from tqdm import tqdm
from transformers import RobertaConfig, RobertaTokenizer
from model.ance import ANCE 
import pytrec_eval

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============ 1. Encoding function ============
def encode(texts, tokenizer, model, device, batch_size=16, max_len=512):
    """Encode text to vectors (supports batch + progress bar)"""
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", ncols=100):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = model(**inputs)
            all_embs.append(embeddings.cpu().numpy())
    return np.vstack(all_embs).astype("float32")


# ============ 2. FAISS index ============
def build_faiss_index(dim, use_gpu=True):
    cpu_index = faiss.IndexFlatIP(dim)  # Inner product similarity
    if use_gpu and torch.cuda.is_available():
        logger.info("Using GPU for FAISS...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        logger.info("Using CPU for FAISS...")
        index = cpu_index
    return index

# ============ 3. Retrieval (dedup, keep highest score only) ============
def search_queries_with_faiss(index, base_query_embeddings, base_pids,
                              test_query_embeddings, topN, query_count):
    """
    topN: Number of candidates returned by faiss retrieval (can be larger than final_topN to ensure enough after dedup)
    topN: Maximum number to keep per query after deduplication
    """
    index.add(base_query_embeddings)
    D, I = index.search(test_query_embeddings, topN * query_count)  # (N_test, topN)

    filtered_D, filtered_ids = [], []

    for d_list, idx_list in zip(D, I):
        # Use dict to record best distance for each pid
        seen = {}
        for d, i in zip(d_list, idx_list):
            pid = base_pids[i]
            if pid not in seen:
                seen[pid] = d
            else:
                # Keep higher distance (higher inner product similarity)
                if d > seen[pid]:
                    seen[pid] = d

        # Sort and take top topN
        sorted_items = sorted(seen.items(), key=lambda x: x[1], reverse=True)[:topN]
        filtered_ids.append([pid for pid, _ in sorted_items])
        filtered_D.append([score for _, score in sorted_items])

    return np.array(filtered_D, dtype="float32"), filtered_ids


# ============ 4. TREC output ============
def output_test_res(test_qids, D, candidate_ids, output_trec_file, run_tag="ance"):
    """
    Write TREC format file: qid Q0 docid rank score run_tag
    """
    with open(output_trec_file, "w") as f:
        for qid, scores, pids in zip(test_qids, D, candidate_ids):
            for rank, (pid, score) in enumerate(zip(pids, scores), start=1):
                # g.write(str(qid) + " Q0 " + str(pid) + " " + str(i + 1) + " " + str(-i - 1 + 200) + ' ' + str(score) + " ance\n")
                f.write(f"{qid} Q0 {pid} {rank} {-rank + 200} {score:.4f} {run_tag}\n")
    logger.info(f"TREC run file saved at {output_trec_file}")


# ============ 5. Evaluation ============
def print_trec_res(run_file, qrel_file, rel_threshold=1):
    with open(run_file, 'r') as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()

    qrels = {}
    qrels_ndcg = {}
    runs = {}

    for line in qrel_data:
        line = line.strip().split("\t")
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}
        qrels_ndcg[query][passage] = rel
        qrels[query][passage] = 1 if rel >= rel_threshold else 0

    for line in run_data:
        line = line.strip().split()
        query = line[0]
        passage = line[2]
        score = float(line[4])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = score

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {"map", "recip_rank", "recall.1", "recall.5", "recall.10", "recall.100", "recall.150", "recall.200"})
    res = evaluator.evaluate(runs)

    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_1_list = [v['recall_1'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_150_list = [v['recall_150'] for v in res.values()]
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
        "Recall@150": round(np.average(recall_150_list)*100, 5),
        "Recall@200": round(np.average(recall_200_list)*100, 5),
    }

    logger.info("---------------------Evaluation results:---------------------")
    logger.info(res)
    return res


# ============ 6. Main pipeline ============
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--new_query_jsonl", type=str, required=True, help="New dataset (jsonl) with {p_id, query}")
    parser.add_argument("--test_file_path", type=str, required=True, help="Test data (jsonl) with at least {sample_id, query}")
    parser.add_argument("--pretrained_encoder_path", type=str, default="ance-msmarco")
    parser.add_argument("--trec_gold_qrel_file_path", type=str, default=None, help="Qrel file for evaluation (optional)")
    parser.add_argument("--output_trec_file", type=str, default="retrieval.trec")
    parser.add_argument("--output_high_topk_path", type=str, default="")
    # parser.add_argument("--qrel_file", type=str, default="")
    parser.add_argument("--dataset", type=str, default="topiocqa")
    
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--query_count", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_gpu", type=bool, default=True)
    
    parser.add_argument("--max_concat_length", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

    # 1. Load model
    # tokenizer = AutoTokenizer.from_pretrained(args.pretrained_encoder_path)
    # model = AutoModel.from_pretrained(args.pretrained_encoder_path).to(device)
    logger.info("Loading model")
    config = RobertaConfig.from_pretrained(args.pretrained_encoder_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_encoder_path, do_lower_case=True)
    model = ANCE.from_pretrained(args.pretrained_encoder_path, config=config).to(device)
    model.eval()

    # 2. Encode base queries
    logger.info("Encoding base queries...")
    base_queries, base_pids = [], []
    with open(args.new_query_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            base_queries.append(obj["query"])
            base_pids.append(str(obj["p_id"]))  # Ensure string type
    base_embeddings = encode(base_queries, tokenizer, model, device,
                             batch_size=args.batch_size, max_len=args.max_concat_length)

    # 3. Encode test queries
    logger.info("Encoding test queries...")
    test_queries, test_qids = [], []
    with open(args.test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # if len(obj["pos_docs_pids"]) > 0:
            # # Combined query
            if args.dataset == "topiocqa":
                # Use dialogue
                ctx_utts_text = obj['cur_utt_text'].strip().split(" [SEP] ")
                cur_utt = ctx_utts_text[-1]
                ctx_utts_text = ctx_utts_text[:-1]
                
                max_query_length = 32
                max_response_length = 64
                
                flat_concat = cur_utt
                for j in range(len(ctx_utts_text) - 1, -1, -1):
                    max_len = max_response_length if j % 2 == 1 else max_query_length
                    utt = ctx_utts_text[j][:max_len]
                    if len(flat_concat) + len(utt) > args.max_concat_length:
                        remaining_length = args.max_concat_length - len(flat_concat)
                        if remaining_length > 0:
                            flat_concat += utt[:remaining_length]
                        break
                    else:
                        flat_concat += utt
                test_queries.append(flat_concat)
                # # Use rewritten query
                # test_queries.append(obj["oracle_utt_text"])
            else:
                cur_query = obj["cur_utt_text"]
                context_query = obj["ctx_utts_text"]  
                test_queries.append("\n".join(context_query + [cur_query]))
                # # Use rewritten query
                # test_queries.append(obj["oracle_utt_text"])
            test_qids.append(str(obj.get("sample_id", len(test_qids))))
    test_embeddings = encode(test_queries, tokenizer, model, device,
                             batch_size=args.batch_size)

    # 4. Build index + retrieve
    index = build_faiss_index(base_embeddings.shape[1], use_gpu=args.use_gpu)
    D, candidate_ids = search_queries_with_faiss(index,
                                                 base_embeddings,
                                                 base_pids,
                                                 test_embeddings,
                                                 args.top_k, query_count=args.query_count)
    # 5. Extract top-k PIDs for each test sample
    logger.info("Extracting top-k PIDs for each test query...")
    high_topk_pid = {}  # {qid: [pid, ...]}

    for qid, cands in zip(test_qids, candidate_ids):
        # cands are already top_k corresponding base_pids
        high_topk_pid[qid] = cands.tolist() if hasattr(cands, "tolist") else list(cands)
    with open(args.output_high_topk_path, "w") as f:
        json.dump(high_topk_pid, f, ensure_ascii=False, indent=4)

    logger.info(f"Finished extracting top-{args.top_k} PIDs for {len(high_topk_pid)} test queries.")


    # 6. Output TREC format
    output_test_res(test_qids, D, candidate_ids, args.output_trec_file,
                    run_tag=os.path.basename(args.pretrained_encoder_path))

    # 7. If qrel available, do evaluation
    if args.trec_gold_qrel_file_path is not None:
        print_trec_res(args.output_trec_file, args.trec_gold_qrel_file_path)


if __name__ == "__main__":
    main()
