import os
import pickle
from json.tool import main
import json
from tqdm import tqdm
import csv
import random


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pload(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    print('load path = {} object'.format(path))
    return res


def pstore(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f)
    print('store object in path = {} ok'.format(path))


def gen_qrecc_passage_collection(input_passage_dir, output_file, pid2rawpid_path):
    '''
    - input_passage_dir = "collection-paragraph"
    - output_file = "qrecc_collection.tsv"
    - pid2rawpid_path = "pid2rawpid.pkl"
    '''

    def process_qrecc_per_dir(dir_path, pid, pid2rawpid, fw):
        filenames = os.listdir(dir_path)
        for filename in tqdm(filenames):
            with open(os.path.join(dir_path, filename), "r") as f:
                data = f.readlines()
            for line in tqdm(data):
                line = json.loads(line)
                raw_pid = line["id"]
                passage = line["contents"]
                pid2rawpid[pid] = raw_pid
                fw.write("{}\t{}".format(pid, passage))
                fw.write("\n")

                pid += 1

        return pid, pid2rawpid

    pdir1 = os.path.join(input_passage_dir, "commoncrawl")
    pdir2 = os.path.join(input_passage_dir, "wayback")
    pdir3 = os.path.join(input_passage_dir, "wayback-backfill")

    pid = 0
    pid2rawpid = {}

    with open(output_file, "w") as fw:
        pid, pid2rawpid = process_qrecc_per_dir(pdir1, pid, pid2rawpid, fw)
        logger.info("{} process ok!".format(pdir1))
        pid, pid2rawpid = process_qrecc_per_dir(pdir2, pid, pid2rawpid, fw)
        logger.info("{} process ok!".format(pdir2))
        pid, pid2rawpid = process_qrecc_per_dir(pdir3, pid, pid2rawpid, fw)
        logger.info("{} process ok!".format(pdir3))

    pstore(pid2rawpid, pid2rawpid_path)

    logger.info("generate QReCC passage collection -> {} ok!".format(output_file))
    logger.info("#totoal passages = {}".format(pid))


def gen_qrecc_qrel(input_test_file, output_qrel_file, pid2rawpid_path):
    '''
    - input_test_file = "scai-qrecc21-test-turns.json"
    - pid2rawpid_path = "pid2rawpid.pkl"
    - output_qrel_file = "qrecc_qrel.tsv"
    '''
    with open(input_test_file, "r") as f:
        data = json.load(f)

    pid2rawpid = pload(pid2rawpid_path)
    rawpid2pid = {}
    for pid, rawpid in pid2rawpid.items():
        rawpid2pid[rawpid] = pid

    with open(output_qrel_file, "w") as f:
        for line in tqdm(data):
            sample_id = "{}_{}_{}".format("QReCC-Test", line['Conversation_no'], line['Turn_no'])
            for rawpid in line['Truth_passages']:
                f.write("{}\t{}\t{}\t{}".format(sample_id, 0, rawpid2pid[rawpid], 1))
                f.write('\n')

    logger.info("generate qrecc qrel file -> {} ok!".format(output_qrel_file))


def gen_qrecc_train_test_files(train_inputfile,
                               test_inputfile,
                               train_outputfile,
                               test_outputfile,
                               pid2rawpid_path,
                               max_random_neg_raito=5):
    '''
    - train_inputfile = "scai-qrecc21-training-turns.json"
    - test_inputfile = "scai-qrecc21-test-turns.json"
    - train_outputfile = "train.json"
    - test_outputfile = "test.json"
    - pid2rawpid_path = "pid2rawpid.pkl"
    '''
    pid2rawpid = pload(pid2rawpid_path)
    rawpid2pid = {}
    for pid, rawpid in pid2rawpid.items():
        rawpid2pid[rawpid] = pid

    sid2utt = {}
    sid2pospid = {}

    # train & test raw files
    num_num_doc = 54573064
    outputfile2inputfile = {train_outputfile: train_inputfile,
                            test_outputfile: test_inputfile}
    for outputfile in outputfile2inputfile:
        with open(outputfile2inputfile[outputfile], "r") as f:
            data = json.load(f)

        with open(outputfile, "w") as f:
            for line in tqdm(data):
                record = {}
                sample_title = "QReCC-Train" if outputfile == train_outputfile else "QReCC-Test"
                sample_id = "{}_{}_{}".format(sample_title, line['Conversation_no'], line['Turn_no'])
                record["sample_id"] = sample_id
                record["source"] = line["Conversation_source"]

                cur_utt_text = line["Question"] if int(line['Turn_no']) != 1 else line[
                    "Truth_rewrite"]  # according to the paper of CONQRR
                sid2utt[sample_id] = cur_utt_text
                record["cur_utt_text"] = cur_utt_text

                oracle_utt_text = line["Truth_rewrite"]
                record["oracle_utt_text"] = oracle_utt_text

                cur_response_text = line["Truth_answer"]
                # record["cur_response_text"] = cur_response_text
                record["answer"] = cur_response_text

                ctx_utts_text = []
                for i in range(0, len(line['Context'])):
                    if i % 2 == 0:
                        ctx_query_utt = sid2utt[
                            "{}_{}_{}".format(sample_title, line['Conversation_no'], int(i / 2) + 1)]
                        ctx_utts_text.append(ctx_query_utt)
                    else:
                        ctx_response_utt = line['Context'][i]
                        ctx_utts_text.append(ctx_response_utt)
                record["ctx_utts_text"] = ctx_utts_text

                # Actually useful for training file only
                # process pos doc info, only store pos docs ids and random negative doc ids.
                # Then we will add neg doc ids and then extract doc content.
                pos_docs_pids = []
                for rawpid in line['Truth_passages']:
                    pos_pid = rawpid2pid[rawpid]
                    pos_docs_pids.append(pos_pid)
                sid2pospid[sample_id] = pos_docs_pids
                record["pos_docs_pids"] = pos_docs_pids

                f.write(json.dumps(record))
                f.write('\n')

    logger.info("QReCC train test file preprocessing (first stage) ok!")


if __name__ == "__main__":
    # input_passage_dir = "data/qrecc/collection-paragraph"
    # output_file = "data/qrecc/new_preprocessed/qrecc_collection_1.tsv"
    # pid2rawpid_path = "data/qrecc/new_preprocessed/pid2rawpid_1.pkl"
    # gen_qrecc_passage_collection(input_passage_dir, output_file, pid2rawpid_path)

    train_inputfile = "data/qrecc/ori_data/scai-qrecc21-training-turns.json"
    test_inputfile = "data/qrecc/ori_data/scai-qrecc21-test-turns.json"
    train_outputfile = "data/qrecc/new_data/train.json"
    test_outputfile = "data/qrecc/new_data/test.json"
    pid2rawpid_path = "data/qrecc/new_preprocessed/pid2rawpid.pkl"
    gen_qrecc_train_test_files(train_inputfile, test_inputfile, train_outputfile, test_outputfile, pid2rawpid_path)

    # Example usage:
    # input_test_file = test_inputfile
    # output_qrel_file = "data/qrecc/qrecc_qrel.tsv"
    # gen_qrecc_qrel(input_test_file, output_qrel_file, pid2rawpid_path)
    
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