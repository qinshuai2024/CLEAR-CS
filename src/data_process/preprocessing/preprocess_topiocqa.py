from json.tool import main
import json
from tqdm import tqdm, trange
import csv
import random

def gen_topiocqa_qrel(raw_dev_file_path, output_qrel_file_path):
    '''
    raw_dev_file_path = "gold_dev.json"
    output_qrel_file_path = "topiocqa_qrel.trec"
    '''
    with open(raw_dev_file_path, "r") as f:
        data = json.load(f)
    
    with open(output_qrel_file_path, "w") as f:
        for line in tqdm(data):
            #sample_id = "{}_{}_{}".format("TopiOCQA-Dev", line["conv_id"], line["turn_id"])
            sample_id = "{}-{}".format(line["conv_id"], line["turn_id"])
            for pos in line["positive_ctxs"]:
                #pid = int(pos["passage_id"]) - 1
                pid = int(pos["passage_id"])
                f.write("{}\t{}\t{}\t{}".format(sample_id, 0, pid, 1))
                f.write('\n')


def gen_train_test_files(raw_train_file_path, raw_dev_file_path, output_train_file_path, ouput_test_file_path, collection_file_path):
    '''
    raw_train_file_path = "gold_train.json"
    raw_dev_file_path = "gold_dev.json"
    output_train_file_path = "train.json"
    ouput_test_file_path = "test.json"
    collection_file_path = "full_wiki_segments.tsv"
    '''
    qid2passage = {}
    with open(collection_file_path, 'r') as fin:
        reader = csv.reader(fin, delimiter="\t")
        for i, row in enumerate(tqdm(reader)):
            if row[0] == "id": # ['id', 'text', 'title'] id begin from 1
                continue
            idx, text, title = int(row[0]), row[1], ' '.join(row[2].split(' [SEP] '))
            qid2passage[idx] = " ".join([title, text])

    with open(raw_train_file_path, "r") as f:
        data = json.load(f)
    
    last_conv_id = -1
    last_response = ""
    context_queries_and_answers = []
    context_pos_docs_pids = set()
    random_pid = list(range(25700592))

    with open(output_train_file_path, "w") as f:
        for line in tqdm(data):
            sample_id = "{}_{}_{}".format("TopiOCQA-Train", line["conv_id"], line["turn_id"])
            # sample_id = "{}-{}".format(line["conv_id"], line["turn_id"])
            query = line["question"]
            answers = line["answers"]
            if len(answers) == 0:
                answer = "UNANSWERABLE"
            else:
                answer = answers[0]

            positive_ctxs = line["positive_ctxs"]
            pos_docs = []
            pos_docs_pids = []
            for pos in positive_ctxs:
                passage = pos["title"].rstrip().replace(' [SEP] ', ' ') + ' ' + pos["text"].rstrip()
                pos_docs.append(passage)
                pos_docs_pids.append(int(pos["passage_id"]))            
            # hard_negative_ctxs = line["hard_negative_ctxs"]
            # negative_ctxs = line["negative_ctxs"]

            record = {}
            record["sample_id"] = sample_id
            record["cur_utt_text"] = query
            record["answer"] = answer
            if int(line["conv_id"]) != last_conv_id:
                context_queries_and_answers = []
                context_pos_docs_pids = set()
                last_response = ""
            #record["ctx_utts_text"] = context_queries_and_answers
            record["last_response"] = last_response
            record["pos_docs"] = pos_docs
            record["pos_docs_pids"] = pos_docs_pids

            # prepos_neg_docs_pids = list(context_pos_docs_pids - set(pos_docs_pids))
            # neg_docs = []
            # neg_docs_pids = []
            # if len(prepos_neg_docs_pids):
            #     neg_docs_pids.append(random.choice(prepos_neg_docs_pids))
            # else:
            #     neg_docs_pids.append(random.choice(random_pid))
            # neg_docs.append(qid2passage[neg_docs_pids[0]])

            # record["neg_docs"] = neg_docs
            # record["neg_docs_pids"] = neg_docs_pids
            # record["prepos_neg_docs_pids"] = prepos_neg_docs_pids
            f.write(json.dumps(record))
            f.write('\n')

            last_response = positive_ctxs[0]["title"].rstrip().replace(' [SEP] ', ' ') + ' ' + positive_ctxs[0]["text"].rstrip()
            context_pos_docs_pids |= set(pos_docs_pids)
            #context_queries_and_answers.append(query)
            #context_queries_and_answers.append(answer)
            last_conv_id = int(line["conv_id"])


    with open(raw_dev_file_path, "r") as f:
        data = json.load(f)
    
    last_conv_id = -1
    last_response = ""
    context_queries_and_answers = []
    with open(ouput_test_file_path, "w") as f:
        for line in tqdm(data):
            sample_id = "{}_{}_{}".format("TopiOCQA-Dev", line["conv_id"], line["turn_id"])
            # sample_id = "{}-{}".format(line["conv_id"], line["turn_id"])
            query = line["question"]
            answers = line["answers"]
            if len(answers) == 0:
                answer = "UNANSWERABLE"
            else:
                answer = answers[0]

            positive_ctxs = line["positive_ctxs"]
            pos_docs = []
            pos_docs_pids = []
            for pos in positive_ctxs:
                passage = pos["title"].rstrip().replace(' [SEP] ', ' ') + ' ' + pos["text"].rstrip()
                pos_docs.append(passage)
                pos_docs_pids.append(int(pos["passage_id"]))
            # hard_negative_ctxs = line["hard_negative_ctxs"]
            # negative_ctxs = line["negative_ctxs"]

            record = {}
            record["sample_id"] = sample_id
            record["cur_utt_text"] = query
            record["answer"] = answer
            if int(line["conv_id"]) != last_conv_id:
                context_queries_and_answers = []
                context_pos_docs_pids = set()
            #record["ctx_utts_text"] = context_queries_and_answers
            # record["last_response"] = last_response
            record["pos_docs"] = pos_docs
            record["pos_docs_pids"] = pos_docs_pids

            # prepos_neg_docs_pids = list(context_pos_docs_pids - set(pos_docs_pids))
            # neg_docs = []
            # neg_docs_pids = []
            # if len(prepos_neg_docs_pids):
            #     neg_docs_pids.append(random.choice(prepos_neg_docs_pids))
            # else:
            #     neg_docs_pids.append(random.choice(random_pid))
            # neg_docs.append(qid2passage[neg_docs_pids[0]])

            # record["neg_docs"] = neg_docs
            # record["neg_docs_pids"] = neg_docs_pids
            # record["prepos_neg_docs_pids"] = prepos_neg_docs_pids
            f.write(json.dumps(record))
            f.write('\n')

            last_response = positive_ctxs[0]["title"].rstrip().replace(' [SEP] ', ' ') + ' ' + positive_ctxs[0]["text"].rstrip()
            context_pos_docs_pids |= set(pos_docs_pids)
            #context_queries_and_answers.append(query)
            #context_queries_and_answers.append(answer)
            last_conv_id = int(line["conv_id"])

def merge_rel_label_info(rel_file, orig_file, new_file):
    # rel_file: train/dev_rel_label_rawq.json
    # orig_file: train/test.json
    # new_file: train/test_with_gold_rel.json
    with open(rel_file, "r") as f:
        rel_labels = f.readlines()

    with open(orig_file, 'r') as f, open(new_file, 'w') as g:
        lines = f.readlines()
        for i in range(len(lines)):
            line_dict = json.loads(lines[i])
            sample_id = line_dict['sample_id']
            if sample_id.split('-')[-1] != '1':
                assert sample_id == json.loads(rel_labels[i])['id']
                rel_label = json.loads(rel_labels[i])['rel_label']
                line_dict['rel_label'] = rel_label
            else:
                line_dict['rel_label'] = []
            json.dump(line_dict, g)
            g.write('\n')


def merge_rewrite(input_file, rewrite_file, output_file):
    "将查询重写合并进入训练数据中"
    with open(rewrite_file, "r") as f:
        rewrite = json.load(f)
    
    with open(input_file, 'r') as f:
        data = f.readlines()
    
    cnt = 0
    with open(output_file, 'w') as f:
        for i, line in enumerate(data):
            line = json.loads(line.strip())
            q = rewrite[i]
            q_id = "{}-{}".format(q["conv_id"], q["turn_id"])
            if q_id == line['sample_id']:
                line['oracle_utt_text'] = q['question']
            else:
                cnt += 1
            f.write(json.dumps(line))
            f.write('\n')
            
    print(f"未匹配数量：{cnt}")
            


if __name__ == "__main__":
    
    raw_train_file_path = "data/topiocqa/ori_data/gold_train.json"
    raw_dev_file_path = "data/topiocqa/ori_data/gold_dev.json"
    output_train_file_path = "data/topiocqa/train.json"
    output_test_file_path = "data/topiocqa/test.json"
    collection_file_path = "data/topiocqa/full_wiki_segments.tsv"
    gen_train_test_files(raw_train_file_path, raw_dev_file_path, output_train_file_path, output_test_file_path, collection_file_path)

    # raw_dev_file_path = "data/topiocqa/gold_dev.json"
    # output_qrel_file_path = "data/topiocqa/topiocqa_qrel.tsv"
    # gen_topiocqa_qrel(raw_dev_file_path, output_qrel_file_path)

    # input_file = 'data/topiocqa/train.json'
    # rewrite_file = 'data/topiocqa/ori_data/train_rewrite.json'
    # output_file = 'data/topiocqa/train_with_rewrite.json'
    # merge_rewrite(input_file, rewrite_file, output_file)
    
    input_file = 'data/topiocqa/test.json'
    rewrite_file = 'data/topiocqa/ori_data/dev_rewrite.json'
    output_file = 'data/topiocqa/test_with_rewrite.json'
    merge_rewrite(input_file, rewrite_file, output_file)

"""
每条样本包含至少这些字段：
sample_id: 唯一ID  TopiOCQA-Train_1_2
cur_utt_text: 当前用户查询
oracle_utt_text: 查询重写（无则设为 cur_utt_text）
answer:当前轮的参考答案        #   cur_response_text: 当前轮的参考答案/响应（TopiOCQA 之前叫 answer）
ctx_utts_text: 历史上下文，按 [q1, a1, q2, a2, ...] 顺序
pos_docs_text: 正样本文档文本列表
pos_docs_pids: 正样本文档id列表
"""