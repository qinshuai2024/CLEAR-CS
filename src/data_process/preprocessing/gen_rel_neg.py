import json
import pickle
from tqdm import tqdm 
import pytrec_eval
import numpy as np
import random
import orjson

# def gen_rel_topk(rel_path, ori_path, rel_10_id_path, rel_10_text_path, qrecc_collection_path, topk=10, lastk=None):
#     """ 将检索到的相关段落id转换为文本, 插入到原始数据中，存放到新文件
    
#     Args:
#         rel_path: 检索得到的rel文件, 存放的是字典 query_id:[(pid, score),...]
#         ori_path: 
#         topk:最高难度的负样本
#         lastk:充当中等难度的负样本
#     """
#     with open(rel_path, "r") as f:
#         qids_to_ranked_candidate_passages = json.load(f)

#     needed_pids_set = set()  # 记录用到的pid
#     rel_topk = {} 
#     for qid, items in qids_to_ranked_candidate_passages.items():
#         rel_topk[qid] = []
#         for p_s in items[:topk]:
#             rel_topk[qid].append(p_s[0])
#             needed_pids_set.add(p_s[0])

#     # 读取test内容
#     test_records = []
#     with open(ori_path, "r") as f:
#         for line in f.readlines():
#             test_records.append(json.loads(line))

#     # 读取段落内容，添加前十个相关段落文本
#     pid2doc = {}   # 存储用到的pid与对应text的对应关系
#     num_num_doc = 54573064
#     for line in tqdm(open(qrecc_collection_path, "r"), total=num_num_doc):
#         try:
#             pid, doc = line.strip().split('\t')
#             pid = int(pid)
#         except:
#             pid = int(line.strip().split('\t')[0])
#             doc = ""  
#         if pid in needed_pids_set:
#             pid2doc[pid] = doc


#     print("文本写入开始")
#     with open(rel_10_text_path, "w") as f:
#         for record in test_records:
#             rel_10_pids = rel_topk[record["sample_id"]]
#             record["rel_10_text"] = [pid2doc[pid] for pid in rel_10_pids]
#             f.write(json.dumps(record) + '\n')


# def cal_queries_no_rel(filepath, name="test"):
#     """计算数据集中没有相关段落的 查询的数量"""
#     # 读取test内容
#     cnt, sum = 0, 0
#     with open(filepath, "r") as f:
#         for line in f.readlines():
#             sum += 1
#             sample = json.loads(line)
#             if len(sample["pos_docs_pids"]) == 0:
#                 cnt += 1
#     print(f"{name}数据集中, 总turns: {sum}, 没有相关段落的查询数: {cnt}, 占比: {cnt/sum:.2%}")



# def filter_rel_neg(rel_10_id_path, output_rel_neg_path, qrecc_collection_path, rel_neg_ratio=2, random_neg_ratio=0):
#     with open(rel_10_id_path, 'r') as f:
#         data = f.readlines()

#     needed_pids_set = set()
#     print("开始选取rel_neg_ids")
#     modified_data = []
#     for line in tqdm(data):
#         line = json.loads(line)
#         cur_doc_id = set(line["pos_docs_pids"]) | set(line["random_neg_docs_pids"][:random_neg_ratio])
#         needed_pids_set = needed_pids_set | cur_doc_id
#         # 选取rel_neg
#         rel_10_ids = line["rel_10_id"]
#         rel_neg = []
#         for id in rel_10_ids:
#             if id not in cur_doc_id and id not in rel_neg:
#                 rel_neg.append(id)
#             if len(rel_neg) == rel_neg_ratio:
#                 break
#         line["rel_neg_ids"] = rel_neg
#         modified_data.append(line)
#         needed_pids_set = needed_pids_set | set(rel_neg)

#     # load collection.tsv
#     pid2doc = {}
#     num_num_doc = 54573064
#     bad_doc_set = set()

#     for line in tqdm(open(qrecc_collection_path, "r"), total=num_num_doc):
#         try:
#             pid, doc = line.strip().split('\t')
#             pid = int(pid)
#         except:
#             pid = int(line.strip().split('\t')[0])
#             doc = ""
#             bad_doc_set.add(pid)
#         if pid in needed_pids_set:
#             pid2doc[pid] = doc

#     # Merge doc content to the train file

#     with open(output_rel_neg_path, 'w') as fw:
#         for line in modified_data:
#             # line = json.loads(line)

#             pos_docs_text = []
#             for pid in line["pos_docs_pids"]:
#                 if pid in pid2doc:
#                     pos_docs_text.append(pid2doc[pid])
                    
#             line["pos_docs_text"] = pos_docs_text

#             if len(pos_docs_text) > 0:
#                 rel_neg_docs_text = []
#                 for pid in line["rel_neg_ids"]:
#                     if pid in pid2doc:
#                         rel_neg_docs_text.append(pid2doc[pid])

#                 line["rel_neg_docs_text"] = rel_neg_docs_text
                
#                 # 如果需要加入random neg
#                 if random_neg_ratio > 0:
#                     random_neg_docs_text = []
#                     for pid in line["random_neg_docs_pids"][:random_neg_ratio]:
#                         if pid in pid2doc:
#                             random_neg_docs_text.append(pid2doc[pid])

#                     line["random_neg_docs_text"] = random_neg_docs_text

#             fw.write(json.dumps(line))
#             fw.write('\n')

#     print("QReCC train file with doc (pos+rel_neg) content are generated OK!")


# def fliter_rel_irrel(rel_10_id_path, output_rel_neg_path, qrecc_collection_path, rel_neg_ratio=1, irrel_neg_ratio=1):
#     """为数据增强两种噪音，相关但不含正确答案和完全不相关

#     Args:
#         - rel_10_id_path: 包含random_doc_ids 和 rel_10_ids
#         - output_rel_neg_path: 输出包含两种噪声文本的json数据
#         - 
#     """
#     with open(rel_10_id_path, 'r') as f:
#         data = f.readlines()

#     needed_pids_set = set()
#     print("开始选取rel_irrel_neg_ids")
#     modified_data = []
#     cnt_no_rel, cnt_no_irrel = 0, 0
#     for line in tqdm(data):
#         line = json.loads(line)
#         # pos_id_1 = line["pos_docs_pids"][0] 
#         cur_doc_id = set(line["pos_docs_pids"])
#         needed_pids_set = needed_pids_set | cur_doc_id
        
#         # 选取rel_neg
#         rel_10_ids = line["rel_10_id"]
#         rel_neg = []
#         for id in rel_10_ids:
#             if id not in cur_doc_id and id not in rel_neg:
#                 rel_neg.append(id)
#             if len(rel_neg) == rel_neg_ratio:
#                 break
#         if len(rel_neg) < rel_neg_ratio:  # 统计有多少个查询不足rel_neg_ratio
#             cnt_no_rel += 1
#         line["rel_neg_ids"] = rel_neg
#         cur_doc_id = cur_doc_id | set(rel_10_ids)   # 把前10个都加进去，更好判别选取irrel段落
#         needed_pids_set = needed_pids_set | set(rel_neg)

#         # 选取irrel_neg
#         random_ids = line["random_neg_docs_pids"]
#         irrel_neg = []
#         for id in random_ids:
#             if id not in cur_doc_id and id not in irrel_neg:
#                 irrel_neg.append(id)
#             if len(irrel_neg) == irrel_neg_ratio:
#                 break
#         if len(irrel_neg) < irrel_neg_ratio:
#             cnt_no_irrel += 1
#         line["irrel_neg_ids"] = irrel_neg

#         modified_data.append(line)
#         needed_pids_set = needed_pids_set | set(irrel_neg)

#     print(f"有{cnt_no_rel}个查询不足{rel_neg_ratio}个相关样本，有{cnt_no_irrel}个查询不足{irrel_neg_ratio}个不相关样本")
#     # 有1个查询不足3个相关样本，有0个查询不足3个不相关样本
#     # load collection.tsv
#     pid2doc = {}
#     num_num_doc = 54573064
#     bad_doc_set = set()

#     for line in tqdm(open(qrecc_collection_path, "r"), total=num_num_doc):
#         try:
#             pid, doc = line.strip().split('\t')
#             pid = int(pid)
#         except:
#             pid = int(line.strip().split('\t')[0])
#             doc = ""
#             bad_doc_set.add(pid)
#         if pid in needed_pids_set:
#             pid2doc[pid] = doc

#     # Merge doc content to the train file

#     with open(output_rel_neg_path, 'w') as fw:
#         for line in modified_data:
#             # line = json.loads(line)

#             pos_docs_text = []
#             for pid in line["pos_docs_pids"]:
#                 if pid in pid2doc:
#                     pos_docs_text.append(pid2doc[pid])
                    
#             line["pos_docs_text"] = pos_docs_text

#             if len(pos_docs_text) > 0:
#                 rel_neg_docs_text = []
#                 for pid in line["rel_neg_ids"]:
#                     rel_neg_docs_text.append(pid2doc[pid])

#                 line["rel_neg_docs_text"] = rel_neg_docs_text
                
#                 # irrel neg
#                 if irrel_neg_ratio > 0:
#                     irrel_neg_text = []
#                     for pid in line["irrel_neg_ids"]:
#                         irrel_neg_text.append(pid2doc[pid])

#                     line["irrel_neg_text"] = irrel_neg_text

#             fw.write(json.dumps(line))
#             fw.write('\n')

#     print("QReCC train file with doc (pos+rel_neg+irrel_neg) content are generated OK!")


# def cnt_margin(rel_10_id_path, model_path):
#     with open(rel_10_id_path, 'r') as f:
#         data = f.readlines()
    
#     for line in data:
#         line = json.loads(line)
#         # pos_id


# def get_neg_sample(rel_id_path, bm25_rel_id_path, train_path, output_neg_path, qrecc_collection_path, hight_rel_neg_ratio=5, bm25_rel_neg_ratio=5, random_neg_ratio=None, dataset="topiocqa"):
#     """为数据增强两种噪音，相关但不含正确答案和完全不相关

#     Args:
#         - rel_id_path: 包含稠密检索的top100数据 
#         - bm25_rel_id_path: 包含BM25 top100数据 
#         - train_path: 数据集
#         - output_neg_path: 输出包含噪声文本的json数据
#     """
#     random.seed(42)
#     with open(rel_id_path, 'r') as f:
#         id_data = json.load(f)
#     with open(bm25_rel_id_path, 'r') as f:
#         id_data_bm25 = {}
#         for line in f:
#            sample = json.loads(line.strip())
#            id_data_bm25.update(sample)
#     # print(id_data_bm25['1-1'])
#     # return
#     if dataset == "topiocqa":
#         num_num_doc = 25700592
#     else:
#         num_num_doc = 54573064
    
#     needed_pids_set = set()
#     print("开始选取负样本")
#     modified_data = []
#     with open(train_path, 'r') as f:
#         for i, line in enumerate(f):
#             if i % 1000 == 0:
#                 print(f"处理第 {i} 条数据")
#             line = json.loads(line.strip())
#             sample_id = line["sample_id"]
#             rel_ids = id_data[sample_id]
#             bm25_rel_ids = id_data_bm25[sample_id]
#             # 0.正样本
#             cur_doc_id = set(line["pos_docs_pids"])
#             needed_pids_set = needed_pids_set | cur_doc_id
            
#             # 1.选取hight_rel_neg
#             if hight_rel_neg_ratio:
#                 topk_ids = rel_ids[:hight_rel_neg_ratio * 2]
#                 random.shuffle(topk_ids)
#                 hight_rel_neg_ids = []
#                 for id in topk_ids:
#                     if id not in cur_doc_id and id not in hight_rel_neg_ids:
#                         hight_rel_neg_ids.append(id)
#                     if len(hight_rel_neg_ids) == hight_rel_neg_ratio:
#                         break

#                 line["hight_rel_neg_ids"] = hight_rel_neg_ids
#                 needed_pids_set = needed_pids_set | set(hight_rel_neg_ids)
#             # print("选取hight_rel_neg done")

#             # 2.选取bm25_rel_neg
#             if bm25_rel_neg_ratio:
#                 bm25_topk_ids = bm25_rel_ids[:bm25_rel_neg_ratio * 2]
#                 random.shuffle(bm25_topk_ids)
#                 bm25_rel_neg_ids = []
#                 for id in bm25_topk_ids:
#                     if id not in cur_doc_id and id not in bm25_rel_neg_ids:
#                         bm25_rel_neg_ids.append(id)
#                     if len(bm25_rel_neg_ids) == bm25_rel_neg_ratio:
#                         break
#                 # if len(low_rel_neg_ids) < irrel_neg_ratio:
#                 #     cnt_no_irrel += 1
#                 line["bm25_rel_neg_ids"] = bm25_rel_neg_ids
#                 needed_pids_set = needed_pids_set | set(bm25_rel_neg_ids)
#                 # print("选取low_rel_neg done")
            
#             # 3.选取random样本   qrecc不需要
#             if random_neg_ratio:
#                 # all_set = all_sample - cur_doc_id - set(rel_ids)
#                 # random_neg_ids = random.sample(list(all_set), random_neg_ratio)
#                 random_neg_ids = []
#                 while len(random_neg_ids) < random_neg_ratio:
#                     rid = random.randint(0, num_num_doc - 1)
#                     if rid not in cur_doc_id and rid not in rel_ids:
#                         random_neg_ids.append(rid)
#                 line["random_neg_ids"] = random_neg_ids
#                 needed_pids_set = needed_pids_set | set(random_neg_ids)
#             # print("选取random样本 done")
            
#             for key in ["neg_docs", "neg_docs_pids", "prepos_neg_docs_pids"]:
#                 line.pop(key, None)

#             modified_data.append(line)
    
#     with open("experiments/base_ance/topiocqa/data/modified_data.json", "w") as f:
#         json.dump(modified_data, f)
    
#     # with open("experiments/base_ance/topiocqa/data/modified_data.json", "r") as f:
#     #     modified_data = json.load(f)
    
#     # print(f"有{cnt_no_rel}个查询不足{rel_neg_ratio}个相关样本，有{cnt_no_irrel}个查询不足{irrel_neg_ratio}个不相关样本")
#     # 有1个查询不足3个相关样本，有0个查询不足3个不相关样本
    
#     # load collection.tsv
#     pid2doc = {}
#     bad_doc_set = set()
#     print(f"Loading QReCC collection, total {num_num_doc} passages...")
#     for line in tqdm(open(qrecc_collection_path, "r"), total=num_num_doc):
#         try:
#             sample = line.strip().split('\t')
#             if dataset=="topiocqa":
#                 pid, doc, title = sample
#                 if pid == "id":
#                     continue
#                 title = " ".join(title.split(' [SEP] '))
#                 doc = " ".join([title, doc.rstrip()])
#             else:
#                 pid, doc = sample
#             pid = int(pid)
#             if pid in needed_pids_set:   # 只保存需要的，且正常有文本的段落
#                 pid2doc[pid] = doc
#         except:
#             pid = int(line.strip().split('\t')[0])
#             doc = ""
#             bad_doc_set.add(pid)
#     print(f"不存在的pid数量 {len(bad_doc_set)}")

#     # Merge doc content to the train file
#     with open(output_neg_path, 'w') as fw:
#         for line in modified_data:
#             pos_docs_pids = line["pos_docs_pids"]
#             # pos_docs_text = []
#             # for pid in line["pos_docs_pids"]:
#             #     pos_docs_text.append(pid2doc[pid])
#             # line["pos_docs_text"] = pos_docs_text

#             if len(pos_docs_pids) > 0:
#                 # 1.选取hight_rel_neg
#                 if hight_rel_neg_ratio:
#                     hight_rel_neg_text = []
#                     for pid in line["hight_rel_neg_ids"]:
#                         hight_rel_neg_text.append(pid2doc[pid])
#                     line["hight_rel_neg_text"] = hight_rel_neg_text
                
#                 # 2.选取bm25_rel_neg
#                 if bm25_rel_neg_ratio:
#                     bm25_rel_neg_text = []
#                     for pid in line["bm25_rel_neg_ids"]:
#                         bm25_rel_neg_text.append(pid2doc[pid])
#                     line["bm25_rel_neg_text"] = bm25_rel_neg_text

#                 if topic_shift_neg_ratio:
#                     topic_shift_neg_text = []
#                     for pid in line["topic_shift_neg_ids"]:
#                         topic_shift_neg_text.append(pid2doc[pid])
#                     line["topic_shift_neg_text"] = topic_shift_neg_text
                
#                 # 3.选取random样本
#                 if random_neg_ratio:
#                     random_neg_text = []
#                     for pid in line["random_neg_ids"]:
#                         random_neg_text.append(pid2doc[pid])
#                     line["random_neg_text"] = random_neg_text

#             fw.write(json.dumps(line))
#             fw.write('\n')

#     print(f"{dataset} train file with doc (pos+neg) content are generated OK!")

def sample_random_negatives_fast(num_num_doc, exclude_set, random_neg_ratio):
    """
    优化2：使用 numpy 向量化随机采样 + 集合过滤
    代替原始 while + random.randint 方案，提速约 10~50x
    """
    sample_size = random_neg_ratio * 5  # 多采一点，防止过滤后不够

    while True:
        # 一次性生成 random_neg_ratio * 5 个候选
        candidates = np.random.randint(0, num_num_doc, size=sample_size)
        # 过滤掉正样本和相关样本
        sampled = [int(x) for x in candidates if x not in exclude_set]
        if len(sampled) >= random_neg_ratio:
            return sampled[:random_neg_ratio]


def get_neg_sample_fast(
    rel_id_path,
    bm25_rel_id_path,
    train_path,
    output_neg_path,
    qrecc_collection_path,
    hight_rel_neg_ratio=5,
    bm25_rel_neg_ratio=5,
    topic_shift_neg=True,
    random_neg_ratio=None,
    dataset="topiocqa"
):
    """为数据增强采样负样本（优化1+2+3版本）
    - 优化1：用集合差集代替循环 in 判断
    - 优化2：随机负样本用 numpy 向量化采样
    - 优化3：I/O 使用 orjson 加速解析
    """
    print(f"处理数据集：{dataset}")
    random.seed(42)
    np.random.seed(42)
    def _parse_sid(sid):
        if dataset == "qrecc":
            toks = sid.split("_")
            if len(toks) >= 2:
                return toks[-2], toks[-1]
            else:
                return sid, None
        elif dataset == "topiocqa":
            toks = sid.split("_")
            if len(toks) >= 2:
                return toks[0], toks[1]
            else:
                return sid, None
        toks = sid.replace("-", "_").split("_")
        if len(toks) >= 2:
            return toks[-2], toks[-1]
        return sid, None

    print("加载稠密检索结果...")
    with open(rel_id_path, "r") as f:
        id_data = json.load(f)

    print("加载BM25检索结果...")
    id_data_bm25 = {}
    with open(bm25_rel_id_path, "r") as f:
        for line in f:
            sample = orjson.loads(line)
            id_data_bm25.update(sample)

    num_num_doc = 25700592 if dataset == "topiocqa" else 54573064
    modified_data = []
    needed_pids_set = set()

    dialog_pos_map = {}
    dialog_turns_map = {}
    dialog_sid_map = {}
    with open(train_path, "r") as f:
        for line in f:
            rec = orjson.loads(line)
            sid = rec["sample_id"]
            d_id, t_id_str = _parse_sid(sid)
            pos_pids = rec.get("pos_docs_pids", [])
            if d_id not in dialog_pos_map:
                dialog_pos_map[d_id] = {}
                dialog_turns_map[d_id] = []
                dialog_sid_map[d_id] = {}
            dialog_pos_map[d_id][t_id_str] = set(pos_pids)
            if t_id_str is not None:
                dialog_sid_map[d_id][t_id_str] = sid
            if t_id_str is not None:
                try:
                    dialog_turns_map[d_id].append(int(t_id_str))
                except:
                    pass

    print("开始采样负样本...")
    with open(train_path, "r") as f:
        for i, line in enumerate(tqdm(f, desc="Processing")):
            line = orjson.loads(line)
            sample_id = line["sample_id"]

            rel_ids = id_data[sample_id]
            bm25_rel_ids = id_data_bm25[sample_id]
            cur_doc_id = set(line["pos_docs_pids"])
            if len(cur_doc_id) == 0:
                continue
            needed_pids_set.update(cur_doc_id)
            used_neg = set(cur_doc_id)

            # 选择topi负样本
            if topic_shift_neg:
                topic_shift_neg_ids = []
                d_id, t_id_str = _parse_sid(sample_id)
                if dataset == "qrecc":
                    pos_map = dialog_pos_map.get(d_id, {})
                    for turn_key, pset in pos_map.items():
                        if t_id_str is not None and turn_key == t_id_str:
                            continue
                        for pid in pset:
                            if pid not in cur_doc_id and pid not in topic_shift_neg_ids:
                                topic_shift_neg_ids.append(pid)
                                if len(topic_shift_neg_ids) >= 4:
                                    break
                        if len(topic_shift_neg_ids) >= 4:
                            break
                    # 回退策略：仅当一个都取不到时，用前后两轮各自 rel_ids 的前2个拼成最多4个
                    if len(topic_shift_neg_ids) < 1:
                        try:
                            t_id = int(t_id_str) if t_id_str is not None else None
                        except:
                            t_id = None
                        if t_id is not None:
                            turns = sorted(set(dialog_turns_map.get(d_id, [])))
                            if t_id in turns:
                                idx = turns.index(t_id)
                                neighbor_turns = []
                                if idx - 1 >= 0:
                                    neighbor_turns.append(turns[idx - 1])
                                if idx + 1 < len(turns):
                                    neighbor_turns.append(turns[idx + 1])

                                exclude_set_fill = set(topic_shift_neg_ids) | used_neg
                                sid_map = dialog_sid_map.get(d_id, {})
                                for nt in neighbor_turns:
                                    nsid = sid_map.get(str(nt))
                                    if nsid is None:
                                        continue
                                    n_rel_ids = id_data.get(nsid, [])
                                    for pid in n_rel_ids[:2]:
                                        try:
                                            pid_int = int(pid)
                                        except:
                                            continue
                                        if pid_int in exclude_set_fill:
                                            continue
                                        topic_shift_neg_ids.append(pid_int)
                                        exclude_set_fill.add(pid_int)
                                        if len(topic_shift_neg_ids) >= 4:
                                            break
                                    if len(topic_shift_neg_ids) >= 4:
                                        break
                else:
                    try:
                        t_id = int(t_id_str) if t_id_str is not None else None
                    except:
                        t_id = None
                    if t_id is not None:
                        turns = sorted(set(dialog_turns_map.get(d_id, [])))
                        if t_id in turns and len(turns) > 1:
                            idx = turns.index(t_id)
                            selected_turns = []
                            if idx == 0:
                                selected_turns = [x for x in turns[1:1+4]]
                            elif idx == len(turns) - 1:
                                start = max(0, len(turns) - 1 - 4)
                                selected_turns = [x for x in turns[start:len(turns)-1]]
                            else:
                                before = [turns[i] for i in range(idx-1, max(-1, idx-3), -1)]
                                after = [turns[i] for i in range(idx+1, min(len(turns), idx+3))]
                                selected_turns = before + after
                                k = 3
                                while len(selected_turns) < 4:
                                    added = False
                                    if idx - k >= 0:
                                        selected_turns.append(turns[idx - k])
                                        added = True
                                    if len(selected_turns) >= 4:
                                        break
                                    if idx + k < len(turns):
                                        selected_turns.append(turns[idx + k])
                                        added = True
                                    if not added:
                                        break
                                    k += 1
                            for ot in selected_turns:
                                pset = dialog_pos_map.get(d_id, {}).get(str(ot), set())
                                for pid in pset:
                                    if pid not in cur_doc_id and pid not in topic_shift_neg_ids:
                                        topic_shift_neg_ids.append(pid)
                line["topic_shift_neg_ids"] = topic_shift_neg_ids
                needed_pids_set.update(topic_shift_neg_ids)
                used_neg.update(topic_shift_neg_ids)

            # === 1️⃣ 高相关负样本 (优化1：集合差集操作) ===
            if hight_rel_neg_ratio:
                rel_ids_set = set(rel_ids)
                candidates = list(rel_ids_set - used_neg)
                # random.shuffle(candidates)
                hight_rel_neg_ids = candidates[:hight_rel_neg_ratio]
                line["hight_rel_neg_ids"] = hight_rel_neg_ids
                needed_pids_set.update(hight_rel_neg_ids)
                used_neg.update(hight_rel_neg_ids)

            # === 2️⃣ BM25相关负样本 (同样用集合差集) ===
            if bm25_rel_neg_ratio:
                bm25_rel_ids_set = set(bm25_rel_ids)
                bm25_candidates = list(bm25_rel_ids_set - used_neg)
                # random.shuffle(bm25_candidates)
                bm25_rel_neg_ids = bm25_candidates[:bm25_rel_neg_ratio]
                bm25_rel_neg_ids = [int(pid) for pid in bm25_rel_neg_ids]
                line["bm25_rel_neg_ids"] = bm25_rel_neg_ids
                needed_pids_set.update(bm25_rel_neg_ids)
                used_neg.update(bm25_rel_neg_ids)

            # === 3️⃣ 随机负样本 (优化2：numpy 向量化采样) ===
            if random_neg_ratio:
                exclude_set = used_neg | set(rel_ids)
                random_neg_ids = sample_random_negatives_fast(
                    num_num_doc, exclude_set, random_neg_ratio
                )
                line["random_neg_ids"] = random_neg_ids
                needed_pids_set.update(random_neg_ids)

            # # 删除无用字段
            # for key in ["neg_docs", "neg_docs_pids", "prepos_neg_docs_pids"]:
            #     line.pop(key, None)

            modified_data.append(line)

    print(f"总样本数: {len(modified_data)}, 共涉及文档数: {len(needed_pids_set)}")
    
    # with open("experiments/base_ance/topiocqa/data/modified_data.json", "w") as f:
    #     json.dump(modified_data, f)
    
    # load collection.tsv
    pid2doc = {}
    bad_doc_set = set()
    print(f"Loading QReCC collection, total {num_num_doc} passages...")
    for line in tqdm(open(qrecc_collection_path, "r"), total=num_num_doc):
        try:
            sample = line.rstrip("\n").split('\t')
            if dataset == "topiocqa":
                if len(sample) < 3:
                    pid_str = sample[0] if len(sample) > 0 else None
                    doc = ""
                else:
                    pid_str, doc, title = sample[0], sample[1], '\t'.join(sample[2:])
                    if pid_str == "id":
                        continue
                    title = " ".join(title.split(' [SEP] '))
                    doc = " ".join([title, doc.rstrip()])
            else:
                pid_str = sample[0] if len(sample) > 0 else None
                doc = '\t'.join(sample[1:]) if len(sample) > 1 else ""
            if pid_str is None:
                continue
            pid = int(pid_str)
            if pid in needed_pids_set:
                if doc is not None:
                    pid2doc[pid] = doc
                else:
                    bad_doc_set.add(pid)
            elif doc == "":
                bad_doc_set.add(pid)
        except:
            try:
                pid = int(line.strip().split('\t')[0])
                bad_doc_set.add(pid)
            except:
                pass
    print(f"不存在的pid数量 {len(bad_doc_set)}")

    # Merge doc content to the train file
    with open(output_neg_path, 'w') as fw:
        for line in modified_data:
            pos_docs_pids = line["pos_docs_pids"]
            # 选取正样本段落
            pos_docs_text = []
            for pid in line["pos_docs_pids"]:
                if pid in pid2doc:
                    pos_docs_text.append(pid2doc[pid])
            line["pos_docs_text"] = pos_docs_text

            if len(pos_docs_pids) > 0:
                # 0.选取topic_shift_neg_text
                if topic_shift_neg:
                    topic_shift_neg_text = []
                    for pid in line["topic_shift_neg_ids"]:
                        if pid in pid2doc:
                            topic_shift_neg_text.append(pid2doc[pid])
                    line["topic_shift_neg_text"] = topic_shift_neg_text

                # 1.选取hight_rel_neg
                if hight_rel_neg_ratio:
                    hight_rel_neg_text = []
                    for pid in line["hight_rel_neg_ids"]:
                        if pid in pid2doc:
                            hight_rel_neg_text.append(pid2doc[pid])
                    line["hight_rel_neg_text"] = hight_rel_neg_text
                
                # 2.选取bm25_rel_neg
                if bm25_rel_neg_ratio:
                    bm25_rel_neg_text = []
                    for pid in line["bm25_rel_neg_ids"]:
                        if pid in pid2doc:
                            bm25_rel_neg_text.append(pid2doc[pid])
                    line["bm25_rel_neg_text"] = bm25_rel_neg_text
                
                # 3.选取random样本
                if random_neg_ratio:
                    random_neg_text = []
                    for pid in line["random_neg_ids"]:
                        if pid in pid2doc:
                            random_neg_text.append(pid2doc[pid])
                    line["random_neg_text"] = random_neg_text

            fw.write(json.dumps(line))
            fw.write('\n')

    print(f"{dataset} train file with doc (pos+neg) content are generated OK!")

def neg_sample():

    
    # topiocqa
    # rel_id_path="experiments/base_ance_cls/data/train_high_top_100_bm25cls.json" # bm25 cls
    # rel_id_path="experiments/base_topiocqa/data/train_high_top_100_qracdr.json"  #  qracdr
    # rel_id_path="experiments/base_topiocqa/data/train_test_high_top_10_qracdr.json"  # 排序器对qracdr训练集排序结果
    # rel_id_path=""
    # bm25_rel_id_path="data/topiocqa/train_bm25_topk.jsonl"
    # train_path="data/topiocqa/train_with_rewrite.json"
    # output_neg_path="data/topiocqa/train_hard_negs.json"
    # qrecc_collection_path="data/topiocqa/full_wiki_segments.tsv"
    # get_neg_sample_fast(rel_id_path, bm25_rel_id_path, train_path, output_neg_path, qrecc_collection_path, hight_rel_neg_ratio=5, bm25_rel_neg_ratio=1, random_neg_ratio=2, topic_shift_neg=True, dataset="topiocqa")


    # qrecc
    rel_id_path="data/qrecc/new_data/neg/high_top_100.json"
    bm25_rel_id_path="data/qrecc/new_preprocessed/qrecc_bm25_top20.jsonl"
    train_path="data/qrecc/new_data/train.json"
    output_neg_path="data/qrecc/new_data/train_with_negs.json"
    qrecc_collection_path="data/qrecc/new_preprocessed/qrecc_collection.tsv"
    get_neg_sample_fast(rel_id_path, bm25_rel_id_path, train_path, output_neg_path, qrecc_collection_path, hight_rel_neg_ratio=5, bm25_rel_neg_ratio=4, random_neg_ratio=4, topic_shift_neg=True, dataset="qrecc")


"""
"""


if __name__ == "__main__":
    # Example usage:
    # test_path = "data/qrecc/test.json"
    # train_path = "data/qrecc/train.json"
    # cal_queries_no_rel(test_path, "test")
    # cal_queries_no_rel(train_path, "train")
    # gen_train_rel_10()
    # gen_train_rel_text()
    # gen_train_rel_irrel_text()
    neg_sample()
