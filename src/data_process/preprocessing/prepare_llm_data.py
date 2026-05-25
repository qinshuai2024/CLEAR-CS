# import json
# from transformers import AutoTokenizer

# # 加载Llama-3分词器（自动带chat_template）
# MODEL_NAME = "data/qrecc/meta-llama/Llama-3.1-8B"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# input_file = "data/qrecc/new_preprocessed/LLM_retrieval/finetuning_data_prepared.jsonl"      # 原始jsonl
# output_file = "data/qrecc/new_preprocessed/LLM_retrieval/finetuning_data_llama3.jsonl"  # 转换后的jsonl

# with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
#     for line in fin:
#         example = json.loads(line)
#         messages = example["messages"]

#         # 用Llama3的chat模板拼接成特殊token格式
#         text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=False  # 不额外加assistant提示
#         )

#         # 写入新的jsonl（只保留转换后的text）
#         fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

# print(f"转换完成！输出文件: {output_file}")




# import json

# def split_jsonl(file_path, n, output_path):
#     # 读取所有行
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     total = len(lines)
#     chunk_size = total // n
#     remainder = total % n

#     start = 0
#     for i in range(n):
#         # 计算每一份的大小，如果有余数，前几份多一个
#         end = start + chunk_size + (1 if i < remainder else 0)
#         chunk_lines = lines[start:end]

#         with open(f'{output_path}_part{i+1}.jsonl', 'w', encoding='utf-8') as f_out:
#             f_out.writelines(chunk_lines)

#         start = end

# # 使用示例
# # 切分为多个jsonl文件
# split_jsonl('data/qrecc/new_preprocessed/LLM_retrieval/train_passages_top10.jsonl', 3, output_path="data/qrecc/new_preprocessed/LLM_retrieval/train_split/train_passages_top10")



    
    
import json
from tqdm import tqdm
import os

def preprocess_for_finetuning(input_filepath, output_filepath):
    """
    读取原始JSONL数据,提取段落和对话上下文
    并将其格式化为适合Llama 3.1微调的messages格式。
    """
    print(f"开始处理文件: {input_filepath}")
    
    # 定义一个固定的、用于此任务的系统提示
    system_prompt = (
        "You are a meticulous AI assistant that functions as a question generator.Your goal is to carefully read the provided paragraph and create a single, clear, and concise question that can be fully answered using only the information in the paragraph."
    )
    
    
    sample_count = 0
    processed_count = 0
    skipped_count = 0
    
    # 使用tqdm来显示进度条
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        
        # 为了获取总行数用于tqdm进度条
        total_lines = sum(1 for line in open(input_filepath, 'r', encoding='utf-8'))
        infile.seek(0) # 重置文件指针

        for line in tqdm(infile, total=total_lines, desc="Processing samples"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"警告: 跳过无法解析的JSON行: {line.strip()}")
                skipped_count += 1
                continue

            # 1. 提取所需字段
            # pos_docs = data.get("pos_docs_text", [])  
            pos_docs = data.get("pos_docs", [])   # for topiocqa
            ctx_utts = data.get("ctx_utts_text", [])
            cur_utt = data.get("cur_utt_text", "")
            rewrite_query = data.get("oracle_utt_text", "")

            # 2. 验证数据有效性
            # 必须要有正样本段落和当前查询才能构成一个有效的训练样本
            if not pos_docs or not cur_utt:
                skipped_count += 1
                continue

            # 3. 组合数据
            
            # 将对话历史和当前查询合并成完整的对话上下文
            # 这是我们希望模型学会生成的内容
            # full_conversation_context = "\n".join(ctx_utts + [cur_utt])
            full_conversation_context = rewrite_query  # 0820 使用人工重写的查询
            
            # 如果有多个段落，分别提取 段落-查询
            for paragraph_text in pos_docs:
                # 4. 构建Llama 3.1的messages格式
                # user部分包含了我们的指令和输入段落
                user_content = (
                    "Read the following paragraph carefully and generate one specific question that can be answered from the paragraph alone.\n\n"
                    f"**Paragraph:**\n{paragraph_text}"
                )
                
                # assistant部分是我们的期望输出
                assistant_content = full_conversation_context

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
                
                # 5. 写入到新文件
                output_record = {"messages": messages}
                outfile.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                processed_count += 1
            sample_count += 1   
            # (可选) 打印第一个处理好的样本以供检查
            if sample_count == 1:
                print("\n--- 第一个处理好的样本预览 ---")
                print(json.dumps(output_record, indent=4, ensure_ascii=False))
                print("------------------------------\n")

    print("\n处理完成 !")
    print(f"成功处理并写入了 {sample_count} 个样本。")
    print(f"总共数据量 {processed_count} 个样本。")
    print(f"跳过了 {skipped_count} 个无效或格式错误的样本。")
    print(f"输出文件已保存至: {output_filepath}")    
    
    """
    处理完成 !
    成功处理并写入了 29596 个样本。
    总共数据量 34675 个样本。
    跳过了 33905 个无效或格式错误的样本。
    输出文件已保存至: data/qrecc/new_preprocessed/LLM_retrieval/finetuning_data_prepared_rq.jsonl
    """


def split_collection_file(collection_path, output_dir, num_splits=10):
    """
    将大的collection文件拆分为多个小文件
    
    Args:
        collection_path: 原始collection文件路径
        output_dir: 输出目录
        num_splits: 拆分成几个文件
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 首先统计总行数
    print(f"统计文件总行数...")
    total_lines = sum(1 for _ in open(collection_path, 'r'))
    print(f"总共 {total_lines} 行")
    
    lines_per_file = total_lines // num_splits + 1
    
    print(f"开始拆分文件，每个文件约 {lines_per_file} 行...")
    
    current_file_idx = 0
    current_line_count = 0
    output_file = None
    
    for line in tqdm(open(collection_path, 'r'), total=total_lines, desc="拆分文件"):
        if current_line_count % lines_per_file == 0:
            if output_file:
                output_file.close()
            output_path = os.path.join(output_dir, f"collection_part_{current_file_idx}.tsv")
            output_file = open(output_path, 'w')
            print(f"\n创建文件: {output_path}")
            current_file_idx += 1
        
        output_file.write(line)
        current_line_count += 1
    
    if output_file:
        output_file.close()
    
    print(f"拆分完成！共生成 {current_file_idx} 个文件")
    return current_file_idx


def combine_query_with_top_passages(topk_id_path, test_path, qrecc_collection_path, query_topk_passages_path, dataset="topiocqa", use_split_files=False, split_dir=None):
    """
    获取测试集top 100段落id, 组合文本
    
    Args:
        use_split_files: 是否使用拆分后的文件（更快）
        split_dir: 拆分文件所在目录，如果use_split_files=True则必须提供
    """
    # 1.读取tok_ids 和 pos_id
    with open(topk_id_path, "r") as f:
        topk_ids = json.load(f)
    
    with open(test_path, "r") as f:
        test_data = f.readlines()
    
    # 2.设置需要的pid,并保存对应文本

    needed_pids_set = set()
    num_sample = 0
    for record in test_data:  
        record = json.loads(record.strip())
        if len(record["pos_docs_pids"]) > 0:
            needed_pids_set |= set(record["pos_docs_pids"])
            qid = record["sample_id"]
            topk_id = topk_ids[qid][:100]
            needed_pids_set |= set(topk_id)
            num_sample += 1
    print(f"共{num_sample}条数据,  {len(needed_pids_set)}个段落") 
            
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
    
    # 使用拆分文件加载（更快）
    if use_split_files:
        if split_dir is None:
            raise ValueError("使用拆分文件时必须提供split_dir参数")
        
        import glob
        split_files = sorted(glob.glob(os.path.join(split_dir, "collection_part_*.tsv")))
        print(f"使用拆分文件模式，共找到 {len(split_files)} 个文件")
        
        for split_file in split_files:
            print(f"\n处理文件: {os.path.basename(split_file)}")
            # 如果已经找到了所有需要的段落，可以提前退出
            if len(pid2doc) >= len(needed_pids_set):
                print(f"已找到所有需要的段落，跳过剩余文件")
                break
            
            # 统计当前文件的行数用于进度条
            file_lines = sum(1 for _ in open(split_file, 'r'))
            
            for line in tqdm(open(split_file, "r"), total=file_lines, desc=f"加载{os.path.basename(split_file)}"):
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
                    # 仅有pid
                    try:
                        pid = int(line.strip().split('\t')[0])
                        doc = ""
                        bad_doc_set.add(pid)
                    except:
                        pass
            
            print(f"当前已找到 {len(pid2doc)}/{len(needed_pids_set)} 个需要的段落")
    
    # 使用原始单个大文件加载（慢）
    else:
        print(f"Loading collection from single file, total {num_num_doc} passages...")
        print("提示: 使用 use_split_files=True 可以大幅提升速度！")
        for line in tqdm(open(qrecc_collection_path, "r"), total=num_num_doc):
            # 如果已经找到了所有需要的段落，可以提前退出
            if len(pid2doc) >= len(needed_pids_set):
                print(f"\n已找到所有需要的段落，提前结束")
                break
            
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
                # 仅有pid
                try:
                    pid = int(line.strip().split('\t')[0])
                    doc = ""
                    bad_doc_set.add(pid)
                except:
                    pass
        
    print("Loadding collection OK! Total bad passages = {}".format(len(bad_doc_set)))
    
    # 3.pid转文本
    new_data = []
    num_pids = 0
    for pid in needed_pids_set:
        if pid in pid2doc:
            record = {}
            record["p_id"] = pid
            record["paragraph"] = pid2doc[pid]
            new_data.append(record)
        else:
            num_pids += 1
    print(f"段落文本提取成功, 共{len(new_data)}个最终段落, 不存在段落数量: {num_pids}") 
    
    # 4.保存新数据
    with open(query_topk_passages_path, "w") as f:
        for record in new_data:
            f.write(json.dumps(record) + '\n')
            
    print("保存成功") 
    
    """
    qrecc
    共409926个段落
    
    topiocqa
    共108908个段落
    """
    

# --- 主程序入口 ---
if __name__ == "__main__":

    # input_file = 'data/qrecc/new_preprocessed/train_rel_irrel_bm25_text.json' 
    # output_file = 'data/qrecc/new_preprocessed/LLM_retrieval/finetuning_data_prepared_rq.jsonl' 
    
    # # topiocqa
    # input_file = 'data/topiocqa/train_with_rewrite.json' 
    # output_file = 'data/topiocqa/finetuning_data_topiocqa_rq.jsonl' 
    
    # preprocess_for_finetuning(input_file, output_file)
    
    # qrecc
    # topk_id_path = "experiments/base/test/train_high_rel_10_id.json"
    # # test_path = "data/qrecc/new_preprocessed/test.json"
    # test_path = "data/qrecc/new_preprocessed/train_with_doc.json"
    # qrecc_collection_path = "data/qrecc/new_preprocessed/qrecc_collection.tsv"
    # query_topk_passages_path = "data/qrecc/new_preprocessed/LLM_retrieval/train_passages_top10.jsonl"
    # combine_query_with_top_passages(topk_id_path, test_path, qrecc_collection_path, query_topk_passages_path)
    
    # topiocqa  
    # topk_id_path = "experiments/base_ance/topiocqa/data/test_high_top_100_ance.json"
    # test_path = "data/topiocqa/test_with_rewrite.json"
    # # test_path = "data/qrecc/new_preprocessed/train_with_doc.json"
    # qrecc_collection_path = "data/topiocqa/full_wiki_segments.tsv"
    # query_topk_passages_path = "data/topiocqa/LLM_retrieval/test_passages_top100.jsonl"
    # combine_query_with_top_passages(topk_id_path, test_path, qrecc_collection_path, query_topk_passages_path)
    
    
    # # 训练集 
    # topk_id_path = "experiments/base_topiocqa/data/train_high_top_10_qracdr.json"
    # test_path = "data/topiocqa/train_with_rewrite.json"
    # qrecc_collection_path = "data/topiocqa/full_wiki_segments.tsv"
    # query_topk_passages_path = "data/topiocqa/LLM_retrieval/train_passages_top10.jsonl"
    # combine_query_with_top_passages(topk_id_path, test_path, qrecc_collection_path, query_topk_passages_path)

    # # cast19
    # topk_id_path = "experiments/cast/cast19/test_qracdr.json"
    # test_path = "data/cast/data_cast19/new_data/cast19_test_topiocqa_with_pos_pids.jsonl"
    # # test_path = "data/qrecc/new_preprocessed/train_with_doc.json"
    # qrecc_collection_path = "data/cast/cast2019.tsv"
    # query_topk_passages_path = "data/cast/data_cast19/new_data/cast19_test_passages.jsonl"
    # combine_query_with_top_passages(topk_id_path, test_path, qrecc_collection_path, query_topk_passages_path, dataset="cast1920")
    
    # cast19 段落文本提取成功, 共15564个最终段落, 不存在段落数量: 382   都是正样本

    # # cast20
    # topk_id_path = "experiments/cast/cast20/test_qracdr.json"
    # test_path = "data/cast/data_cast20/new_data/cast20_test_topiocqa_with_pos_pids.jsonl"
    # # test_path = "data/qrecc/new_preprocessed/train_with_doc.json"
    # qrecc_collection_path = "data/cast/cast2019.tsv"
    # query_topk_passages_path = "data/cast/data_cast20/new_data/cast20_test_passages.jsonl"
    # combine_query_with_top_passages(topk_id_path, test_path, qrecc_collection_path, query_topk_passages_path, dataset="cast1920")
    
    # cast20 段落文本提取成功, 共15710个最终段落, 不存在段落数量: 326

    # cast21
    topk_id_path = "experiments/cast/cast21/cast21_qrecc_ance.json"
    test_path = "data/cast/data_cast21/new_data/cast21_test_topiocqa_with_pos_pids.jsonl"
    # test_path = "data/qrecc/new_preprocessed/train_with_doc.json"
    qrecc_collection_path = "data/cast/data_cast21/ori_data/passages.tsv"
    query_topk_passages_path = "data/cast/data_cast21/new_data/cast21_test_passages_1.jsonl"
    
    # 方法1: 第一次运行 - 先拆分文件（只需运行一次）
    # split_dir = "data/cast/data_cast21/ori_data/passages_split"
    # split_collection_file(qrecc_collection_path, split_dir, num_splits=10)
    
    # 方法2: 使用拆分后的文件（快速）
    # split_dir = "data/cast/data_cast21/ori_data/passages_split"
    # combine_query_with_top_passages(topk_id_path, test_path, qrecc_collection_path, 
    #                                 query_topk_passages_path, dataset="cast21",
    #                                 use_split_files=True, split_dir=split_dir)
    
    # 方法3: 使用原始文件（慢，但有优化：找到所有段落后会提前退出）
    combine_query_with_top_passages(topk_id_path, test_path, qrecc_collection_path, 
                                    query_topk_passages_path, dataset="cast21")
    

""" 分析LLM检索的结果 能多召回多少"""
# import json

# no_pos_ori_path = "experiments/base/LLM_retrieval/test/high_topk_text_no_pos.json"
# no_pos_llm_path = "experiments/base/LLM_retrieval/test/high_topk_text_no_pos_llm_rq_all.json"

# no_pos_ori_qid = set()
# no_pos_llm_qid = set()
# with open(no_pos_ori_path, "r") as f:
#     for lines in f.readlines():
#         record = json.loads(lines.strip())
#         no_pos_ori_qid.add(record["sample_id"])
        
# with open(no_pos_llm_path, "r") as f:
#     for lines in f.readlines():
#         record = json.loads(lines.strip())
#         no_pos_llm_qid.add(record["sample_id"])
        
# count_s1_not_in_s2 = len(no_pos_ori_qid - no_pos_llm_qid) # 或者 len(set1.difference(set2))
# count_s2_not_in_s1 = len(no_pos_llm_qid - no_pos_ori_qid) # 或者 len(set2.difference(set1))

# print(f"len no_pos_ori_qid {len(no_pos_ori_qid)}")
# print(f"len no_pos_llm_qid {len(no_pos_llm_qid)}")



# print(f"no_pos_ori_qid 中有 {count_s1_not_in_s2} 个字符串不在 no_pos_llm_qid 中")
# print(f"no_pos_llm_qid 中有 {count_s2_not_in_s1} 个字符串不在 no_pos_ori_qid 中")