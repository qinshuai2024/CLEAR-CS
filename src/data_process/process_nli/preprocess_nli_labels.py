import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import orjson
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ================================================================
# 初始化模型
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if device.type == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("启用TF32与cudnn.benchmark以加速推理")
    except Exception as e:
        pass

model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
print("开始加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()
if device.type == "cuda":
    model.half()  # 使用半精度加速推理
    model = torch.compile(model)  # PyTorch 2.0+ 动态编译优化
print("模型加载成功。")

# 提前缓存 label 索引
id2label = model.config.id2label
label2id = {v.lower(): k for k, v in id2label.items()}
entail_id = label2id["entailment"]
neutral_id = label2id.get("neutral", None)
contradiction_id = label2id.get("contradiction", None)

# ================================================================
# 批量获取 entailment 概率
# ================================================================
def get_entail_scores(passages: list[str], answer: str, batch_size: int = 16, return_all_probs: bool = False) -> list:
    """对一批 (passage, answer) 对计算 entailment 概率
    
    Args:
        passages: 段落列表
        answer: 答案文本
        batch_size: 批处理大小
        return_all_probs: 若为 True，返回 [entailment, neutral, contradiction] 三个概率；
                         若为 False，仅返回 entailment 概率（默认）
    
    Returns:
        若 return_all_probs=False: list[float] - 每个段落的蕴涵概率
        若 return_all_probs=True: list[list[float]] - 每个段落的 [蕴涵, 中立, 矛盾] 概率
    """
    scores = []

    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        inputs = tokenizer(
            batch,
            [answer] * len(batch),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            logits = model(**inputs).logits
            
            probs = softmax(logits, dim=-1)
            
            if return_all_probs:
                # 返回 [entailment, neutral, contradiction] 三个概率
                batch_scores = []
                for j in range(probs.shape[0]):
                    three_probs = [
                        probs[j, entail_id].item(),
                        probs[j, neutral_id].item() if neutral_id is not None else 0.0,
                        probs[j, contradiction_id].item() if contradiction_id is not None else 0.0
                    ]
                    batch_scores.append(three_probs)
                scores.extend(batch_scores)
            else:
                # 仅返回 entailment 概率
                scores.extend(probs[:, entail_id].cpu().tolist())

    return scores

def process_sample_topiocqa(sample: dict, return_all_probs=False):
    """对单条 TopiOCQA 样本生成 entail scores"""
    answer = sample["answer"]
    
    # 如果答案是 UNANSWERABLE，则直接赋固定值
    if answer == "UNANSWERABLE":
        if "pos_docs_text" in sample:
            sample["pos_docs_entail"] = [1.0] * len(sample["pos_docs_text"])
        if "hight_rel_neg_text" in sample:
            sample["hight_rel_neg_entail"] = [0.0] * len(sample["hight_rel_neg_text"])
        if "bm25_rel_neg_text" in sample:
            sample["bm25_rel_neg_entail"] = [0.0] * len(sample["bm25_rel_neg_text"])
        if "random_neg_text" in sample:
            sample["random_neg_entail"] = [0.0] * len(sample["random_neg_text"])
        if "topic_shift_neg_text" in sample:
            sample["topic_shift_neg_entail"] = [0.0] * len(sample["topic_shift_neg_text"])
        return sample
    
    if "pos_docs_text" in sample:
        sample["pos_docs_entail"] = get_entail_scores(sample["pos_docs_text"], answer, return_all_probs=return_all_probs)
    if "hight_rel_neg_text" in sample:
        sample["hight_rel_neg_entail"] = get_entail_scores(sample["hight_rel_neg_text"], answer, return_all_probs=return_all_probs)
    if "bm25_rel_neg_text" in sample:
        sample["bm25_rel_neg_entail"] = get_entail_scores(sample["bm25_rel_neg_text"], answer, return_all_probs=return_all_probs)
    if "random_neg_text" in sample:
        sample["random_neg_entail"] = get_entail_scores(sample["random_neg_text"], answer, return_all_probs=return_all_probs)
    if "topic_shift_neg_text" in sample:
        sample["topic_shift_neg_entail"] = get_entail_scores(sample["topic_shift_neg_text"], answer, return_all_probs=return_all_probs)
    return sample

def process_sample_topiocqa_fast(sample: dict, return_all_probs=False, batch_size: int = 32):
    answer = sample["answer"]
    if answer == "UNANSWERABLE":
        if "pos_docs_text" in sample:
            sample["pos_docs_entail"] = [1.0] * len(sample["pos_docs_text"])
        if "hight_rel_neg_text" in sample:
            sample["hight_rel_neg_entail"] = [0.0] * len(sample["hight_rel_neg_text"])
        if "bm25_rel_neg_text" in sample:
            sample["bm25_rel_neg_entail"] = [0.0] * len(sample["bm25_rel_neg_text"])
        if "random_neg_text" in sample:
            sample["random_neg_entail"] = [0.0] * len(sample["random_neg_text"])
        if "topic_shift_neg_text" in sample:
            sample["topic_shift_neg_entail"] = [0.0] * len(sample["topic_shift_neg_text"])
        return sample
    fields = [
        ("pos_docs_text", "pos_docs_entail"),
        ("hight_rel_neg_text", "hight_rel_neg_entail"),
        ("bm25_rel_neg_text", "bm25_rel_neg_entail"),
        ("random_neg_text", "random_neg_entail"),
        ("topic_shift_neg_text", "topic_shift_neg_entail"),
    ]
    unique_passages = []
    passage2idx = {}
    per_field_indices = {}

    for src_field, tgt_field in fields:
        if src_field not in sample:
            continue
        per_field_indices[src_field] = []
        for p in sample[src_field]:
            if p not in passage2idx:
                passage2idx[p] = len(unique_passages)
                unique_passages.append(p)
            per_field_indices[src_field].append(passage2idx[p])

    if len(unique_passages) == 0:
        return sample

    uniq_scores = get_entail_scores(unique_passages, answer, batch_size=batch_size, return_all_probs=return_all_probs)
    for src_field, tgt_field in fields:
        if src_field not in sample:
            continue
        idxs = per_field_indices[src_field]
        if return_all_probs:
            sample[tgt_field] = [uniq_scores[i] for i in idxs]
        else:
            sample[tgt_field] = [float(uniq_scores[i]) for i in idxs]

    return sample

# ================================================================
# 预处理主流程
# ================================================================

def preprocess_dataset(input_file=None, output_file=None, return_all_probs=True, batch_size: int = 32):
    if input_file is None:
        input_file = "data/topiocqa/train_bm25_hard_negs_topi.json"
    if output_file is None:
        output_file = "data/topiocqa/train_bm25_hard_negs_top_entail.json"  # entail

    # pos
    # passage = "Royal Military College, Duntroon Programs The leadership and military training provided at ADFA during the three years of training is considered the equivalent of III Class at RMC. The college also oversees the program for training officers in the Australian Army Reserve. Upon appointment to the Reserves, members join a University Regiment within their location and then undertake their training over the course of five modules run by the various University Regiments around Australia. Additionally, they are required to parade at their unit one night a week and one weekend a month. The final six-and-a-half-week module of the Reserve officer course is conducted at Duntroon."

    print(f"开始处理文件：{input_file}")
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Processing NLI"):
            sample = orjson.loads(line)
            processed = process_sample_topiocqa_fast(sample, return_all_probs=return_all_probs, batch_size=batch_size)
            fout.write(orjson.dumps(processed).decode("utf-8") + "\n")
    print(f"处理完成，结果保存在 {output_file}")

def preprocess_split(base_input, num_splits: int = 4):
    print(f"开始拆分文件：{base_input} -> {num_splits} 份")

    total_lines = 0
    with open(base_input, "r", encoding="utf-8") as fin:
        for _ in fin:
            total_lines += 1
    print(f"总行数{total_lines}")
    lines_per = (total_lines + num_splits - 1) // num_splits
    part_inputs = [f"{base_input}.part{i}" for i in range(num_splits)]

    with open(base_input, "r", encoding="utf-8") as fin:
        writers = [open(p, "w", encoding="utf-8") for p in part_inputs]
        try:
            idx = 0
            shard = 0
            for line in fin:
                if idx >= lines_per and shard < num_splits - 1:
                    shard += 1
                    idx = 0
                writers[shard].write(line)
                idx += 1
        finally:
            for w in writers:
                w.close()
    print(f"拆分成功：{part_inputs[0]}")

def preprocess_merge(final_output, num_splits):
    part_outputs = [f"{final_output}.part{i}" for i in range(num_splits)]
    print("开始合并输出文件...")
    with open(final_output, "w", encoding="utf-8") as fout:
        for i in range(num_splits):
            with open(part_outputs[i], "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.endswith("\n"):
                        fout.write(line)
                    else:
                        fout.write(line + "\n")
    print(f"合并处理完成，结果保存在 {final_output}")

# ================================================================
# 主函数入口
# ================================================================
if __name__ == "__main__":

    # qrecc
    # input_file = "data/qrecc/new_data/train_with_negs.json"
    # output_file = "data/qrecc/new_data/nli/train_with_negs_entail.json"
    # num_splits = 4
    # preprocess_split(base_input=input_file, num_splits=num_splits)

    # input_file = "data/qrecc/new_data/train_with_negs.json.part3"
    # output_file = "data/qrecc/new_data/nli/train_with_negs_entail.json.part3"
    # preprocess_dataset(input_file, output_file, return_all_probs=False)
    
    # input_file = "data/qrecc/new_data/train_with_negs.json"
    # output_file = "data/qrecc/new_data/nli/train_with_negs_entail.json"
    # num_splits = 4
    # preprocess_merge(final_output=output_file, num_splits=num_splits)
    
    
    # topiocqa
    # input_file = "data/qrecc/new_data/train_with_negs.json"
    # output_file = "data/qrecc/new_data/nli/train_with_negs_entail.json"
    # num_splits = 4
    # preprocess_split(base_input=input_file, num_splits=num_splits)

    # input_file = "data/qrecc/new_data/train_with_negs.json.part3"
    # output_file = "data/qrecc/new_data/nli/train_with_negs_entail.json.part3"
    # preprocess_dataset(input_file, output_file, return_all_probs=False)
    
    input_file = "data/qrecc/new_data/train_with_negs.json"
    output_file = "data/qrecc/new_data/nli/train_with_negs_entail.json"
    num_splits = 4
    preprocess_merge(final_output=output_file, num_splits=num_splits)

    # # input_file = "data/qrecc/new_preprocessed/LLM_retrieval/infer_passages_top100.jsonl"
    # # num_splits = 2
    # # preprocess_split(base_input=input_file, num_splits=num_splits)
    # output_file = "data/qrecc/new_preprocessed/LLM_retrieval/infer_split/generated_queries_num7.jsonl"
    # preprocess_merge(final_output=output_file, num_splits=2)

# CUDA_VISIBLE_DEVICES=4 nohup python src/data_process/process_nli/preprocess_nli_labels.py > data/qrecc/new_data/nli/gen_entail_qrecc_3.out 2>&1 &
