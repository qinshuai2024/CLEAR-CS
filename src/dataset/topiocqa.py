import json
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from dataset.data import padding_seq_to_same_length

class TopiocqaDataset(Dataset):
    def __init__(self, args, tokenizer, filename, rewrite_file=None, is_training=False):
        self.args = args
        self.tokenizer = tokenizer
        self.is_training = is_training

        with open(filename, encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

        n = len(self.data)
        n = int(args.use_data_percent * n)
        if n < len(self.data):
            random.seed(args.seed)
            self.data = random.sample(self.data, n)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        args = self.args
        tokenizer = self.tokenizer

        ctx_utts_text = record['cur_utt_text'].strip().split(" [SEP] ")
        cur_utt_text = ctx_utts_text[-1]
        ctx_utts_text = ctx_utts_text[:-1]
        oracle_utt_text = record["oracle_utt_text"]

        pos_docs_text = record["pos_docs"]
        

        # --- encode ---
        cur_utt = tokenizer.encode(
            cur_utt_text, add_special_tokens=True,
            max_length=args.max_query_length, truncation=True
        )

        flat_concat = cur_utt
        for j in range(len(ctx_utts_text) - 1, -1, -1):
            max_len = args.max_response_length if j % 2 == 1 else args.max_query_length
            utt = tokenizer.encode(
                ctx_utts_text[j], add_special_tokens=True,
                max_length=max_len, truncation=True
            )
            if len(flat_concat) + len(utt) > args.max_concat_length:
                flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]
                break
            else:
                flat_concat.extend(utt)

        flat_concat, flat_concat_mask = padding_seq_to_same_length(
            flat_concat, max_pad_length=args.max_concat_length
        )
        sample = [
            record['sample_id'],
            flat_concat, flat_concat_mask,
        ]
        if self.is_training:
            oracle_utt = tokenizer.encode(
                oracle_utt_text, add_special_tokens=True,
                max_length=args.max_query_length, truncation=True
            )
            oracle_utt, oracle_utt_mask = padding_seq_to_same_length(
                oracle_utt, max_pad_length=args.max_query_length
            )
            pos_docs = tokenizer.encode(
                random.choice(pos_docs_text), add_special_tokens=True,
                max_length=args.max_doc_length, truncation=True
            )
            pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length=args.max_doc_length)
            
            sample.append(pos_docs)
            sample.append(pos_docs_mask)
            sample.append(oracle_utt)
            sample.append(oracle_utt_mask)

            if 'bm25_rel_neg_text' in record:
                neg_pool = record['bm25_rel_neg_text']
            else:
                neg_pool = record['neg_docs']
            neg_docs = tokenizer.encode(
                random.choice(neg_pool), add_special_tokens=True,
                max_length=args.max_doc_length, truncation=True
            )
            neg_docs, neg_docs_mask = padding_seq_to_same_length(neg_docs, max_pad_length=args.max_doc_length)
            sample.append(neg_docs)
            sample.append(neg_docs_mask)
        
        return sample

    @staticmethod
    def get_collate_fn(args):
        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_ids": [],
                "bt_conv_qa": [],
                "bt_conv_qa_mask": [],
                "bt_pos_docs": [],
                "bt_pos_docs_mask": [],
                "bt_oracle_utt": [],
                "bt_oracle_utt_mask": [],
            }
            if len(batch[0]) > 3:
                collated_dict["bt_pos_docs"]= []
                collated_dict["bt_pos_docs_mask"]= []
                collated_dict["bt_oracle_utt"]= []
                collated_dict["bt_oracle_utt_mask"]= []
                collated_dict["bt_neg_docs"]= []
                collated_dict["bt_neg_docs_mask"]= []
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_conv_qa"].append(example[1])
                collated_dict["bt_conv_qa_mask"].append(example[2])
                if len(example) > 3:
                    collated_dict["bt_pos_docs"].append(example[3])
                    collated_dict["bt_pos_docs_mask"].append(example[4])
                    collated_dict["bt_oracle_utt"].append(example[5])
                    collated_dict["bt_oracle_utt_mask"].append(example[6])
                    collated_dict["bt_neg_docs"].append(example[7])
                    collated_dict["bt_neg_docs_mask"].append(example[8])
                # print("len(example):", len(example))

            not_need_to_tensor = {"bt_sample_ids"}
            for key in collated_dict:
                if key not in not_need_to_tensor:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn
    
class RerankerNLI4WayDataset(Dataset):
    def __init__(self, file_path, reranker_tokenizer,
                 num_hard_negatives=1,
                 max_query_length=32, max_response_length=64,
                 max_concat_length=256, max_doc_length=512,
                 include_context=True, is_training=True, dataset="topiocqa",
                 bm25_per_sample=2, topic_per_sample=3, random_per_sample=4):
        """
        New Reranker dataset: 4-class + NLI distillation labels
        Label definition:
          0: positive samples pos_docs
          1: similar to current question (hight_rel_neg_text + bm25_rel_neg_text)
          2: similar to dialogue history (topic_shift_neg_text)
          3: completely irrelevant (random_neg_text)
        """
        self.reranker_tokenizer = reranker_tokenizer
        self.num_hard_negatives = num_hard_negatives
        self.max_query_length = max_query_length
        self.max_response_length = max_response_length
        self.max_concat_length = max_concat_length
        self.max_doc_length = max_doc_length
        self.include_context = include_context
        self.is_training = is_training
        self.dataset = dataset
        self.bm25_per_sample = bm25_per_sample
        self.topic_per_sample = topic_per_sample
        self.random_per_sample = random_per_sample
        self.data = self._load_data(file_path)

    def __len__(self):
        return len(self.data)

    def _load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        valid_data = []
        for line in tqdm(data):
            sample = json.loads(line.strip())
            pos_docs = sample.get("pos_docs_text", sample.get("pos_docs", []))
            if len(pos_docs) > 0:
                valid_data.append(sample)
        print(f"Loaded {len(valid_data)} valid samples for 4-way dataset")
        return valid_data

    def _build_concat_query(self, sample):
        if self.dataset == "qrecc":
            cur_utt = sample['cur_utt_text'].strip()
            ctx_utts_text = sample.get('ctx_utts_text', [])
        else:
            ctx_utts_text = sample['cur_utt_text'].strip().split(" [SEP] ")
            cur_utt = ctx_utts_text[-1]
            ctx_utts_text = ctx_utts_text[:-1]

        cur_ids = self.reranker_tokenizer.encode(
            cur_utt, add_special_tokens=True,
            max_length=self.max_query_length, truncation=True
        )
        concat_query = cur_ids.copy()

        for j in range(len(ctx_utts_text) - 1, -1, -1):
            is_response = j % 2 == 1
            max_len = self.max_response_length if is_response else self.max_query_length
            utt = self.reranker_tokenizer.encode(
                ctx_utts_text[j].strip(), add_special_tokens=True,
                max_length=max_len, truncation=True
            )
            if len(concat_query) + len(utt) > self.max_concat_length:
                remaining = self.max_concat_length - len(concat_query) - 1
                if remaining > 0:
                    concat_query += utt[:remaining] + [utt[-1]]
                break
            else:
                concat_query.extend(utt)
        return concat_query[:self.max_concat_length]

    def _encode_text(self, text, max_doc_length, query_len):
        max_doc_len = min(max_doc_length, 512 - query_len)
        return self.reranker_tokenizer.encode(
            text, add_special_tokens=True,
            max_length=max_doc_len, truncation=True
        )

    def __getitem__(self, index):
        sample = self.data[index]
        concat_query = self._build_concat_query(sample)

        # Positive sample
        pos_list = sample.get('pos_docs_text', sample.get('pos_docs', []))
        pos_idx = random.sample(range(len(pos_list)), 1)[0]
        pos_doc = pos_list[pos_idx]
        pos_entails = sample.get('pos_docs_entail', [0.0] * len(pos_list))
        pos_score = float(pos_entails[pos_idx]) if pos_idx < len(pos_entails) else 0.0

        # High relevance (similar to current question)
        high_list = sample.get('hight_rel_neg_text', [])
        high_entails = sample.get('hight_rel_neg_entail', [0.0] * len(high_list))
        # bm25 pool will be used later but also serves as a fallback to fill class-1 shortage
        bm25_list = sample.get('bm25_rel_neg_text', [])
        bm25_entails = sample.get('bm25_rel_neg_entail', [0.0] * len(bm25_list))

        # target number of high-related items
        target_high = self.num_hard_negatives
        # start with high_list shuffled
        high_pool = high_list.copy()
        if self.is_training:
            random.shuffle(high_pool)
        high_sel = high_pool[:min(len(high_pool), target_high)]
        # fill shortage with bm25 items (still class-1)
        if len(high_sel) < target_high:
            need = target_high - len(high_sel)
            if len(bm25_list) == 0:
                bm25_fill_idx = []
            elif len(bm25_list) < need:
                repeats = (need + len(bm25_list) - 1) // len(bm25_list)
                bm25_fill_idx = (list(range(len(bm25_list))) * repeats)[:need]
            else:
                bm25_fill_idx = random.sample(range(len(bm25_list)), need)
            high_sel.extend([bm25_list[i] for i in bm25_fill_idx])

        # compute scores matched by origin (prefer high_list, else bm25)
        high_scores = []
        for d in high_sel:
            if d in high_list:
                i = high_list.index(d)
                s = float(high_entails[i]) if i < len(high_entails) else 0.0
            else:
                try:
                    j = bm25_list.index(d)
                except ValueError:
                    j = -1
                s = float(bm25_entails[j]) if 0 <= j < len(bm25_entails) else 0.0
            high_scores.append(s)

        # BM25 related (similar to current question)
        if len(bm25_list) < self.bm25_per_sample:
            bm25_sel_idx = list(range(len(bm25_list))) * max(1, (self.bm25_per_sample + max(len(bm25_list), 1) - 1) // max(len(bm25_list), 1))
            bm25_sel_idx = bm25_sel_idx[:self.bm25_per_sample]
        else:
            bm25_sel_idx = random.sample(range(len(bm25_list)), self.bm25_per_sample)
        bm25_sel = [bm25_list[i] for i in bm25_sel_idx]
        bm25_scores = [float(bm25_entails[i]) if i < len(bm25_entails) else 0.0 for i in bm25_sel_idx]

        # Topic shift (similar to dialogue history)
        topic_list = sample.get('topic_shift_neg_text', [])
        topic_entails = sample.get('topic_shift_neg_entail', [0.0] * len(topic_list))
        if len(topic_list) < self.topic_per_sample:
            topic_sel_idx = list(range(len(topic_list))) * max(1, (self.topic_per_sample + max(len(topic_list), 1) - 1) // max(len(topic_list), 1))
            topic_sel_idx = topic_sel_idx[:self.topic_per_sample]
        else:
            topic_sel_idx = random.sample(range(len(topic_list)), self.topic_per_sample)
        topic_sel = [topic_list[i] for i in topic_sel_idx]
        topic_scores = [float(topic_entails[i]) if i < len(topic_entails) else 0.0 for i in topic_sel_idx]

        # Random irrelevant
        rand_list = sample.get('random_neg_text', [])
        rand_entails = sample.get('random_neg_entail', [0.0] * len(rand_list))
        if len(rand_list) < self.random_per_sample:
            rand_sel_idx = list(range(len(rand_list))) * max(1, (self.random_per_sample + max(len(rand_list), 1) - 1) // max(len(rand_list), 1))
            rand_sel_idx = rand_sel_idx[:self.random_per_sample]
        else:
            rand_sel_idx = random.sample(range(len(rand_list)), self.random_per_sample)
        rand_sel = [rand_list[i] for i in rand_sel_idx]
        rand_scores = [float(rand_entails[i]) if i < len(rand_entails) else 0.0 for i in rand_sel_idx]

        # Aggregate
        docs = [pos_doc] + high_sel + bm25_sel + topic_sel + rand_sel
        nli_scores = [pos_score] + high_scores + bm25_scores + topic_scores + rand_scores
        labels = [0] + [1] * len(high_sel) + [1] * len(bm25_sel) + [2] * len(topic_sel) + [3] * len(rand_sel)

        # Encode (query + doc)
        query_len = len(concat_query)
        doc_ids = [self._encode_text(doc, self.max_doc_length, query_len) for doc in docs]
        combined_ids = [concat_query + d[1:] for d in doc_ids]

        return {
            "combined": combined_ids,
            "labels": labels,
            "nli_scores": nli_scores,
            "sample_id": sample.get('sample_id', str(index))
        }

    @staticmethod
    def get_collate_fn(pad_token_id=0, global_max_length=512):
        def collate_fn(batch):
            sample_ids = [it["sample_id"] for it in batch]

            # Compute max number of documents in batch
            lens = [len(it["combined"]) for it in batch]
            num_ctx = max(lens)

            def pad_ids(seq, max_len):
                seq = seq[:max_len]
                return seq + [pad_token_id] * (max_len - len(seq))

            combined_tensor_list = []   # [B, num_ctx, L]
            combined_mask_list = []     # [B, num_ctx, L]
            labels_tensor_list = []     # [B, num_ctx]
            nli_tensor_list = []        # [B, num_ctx]
            doc_mask_list = []          # [B, num_ctx]

            for it in batch:
                cur_combined = [pad_ids(c, global_max_length) for c in it["combined"]]
                cur_labels = it["labels"]
                cur_nli = it["nli_scores"]

                cur_len = len(cur_combined)

                if cur_len < num_ctx:
                    pad_doc = [pad_token_id] * global_max_length
                    cur_combined += [pad_doc] * (num_ctx - cur_len)
                    cur_labels += [-100] * (num_ctx - cur_len)
                    cur_nli += [0.0] * (num_ctx - cur_len)

                cur_combined_tensor = torch.LongTensor(cur_combined)  # [num_ctx, L]
                cur_mask_tensor = (cur_combined_tensor != pad_token_id).long()  # [num_ctx, L]
                cur_labels_tensor = torch.LongTensor(cur_labels)  # [num_ctx]
                cur_nli_tensor = torch.FloatTensor(cur_nli)       # [num_ctx]
                cur_doc_mask = torch.zeros(num_ctx, dtype=torch.long)
                cur_doc_mask[:len(it["combined"])] = 1

                combined_tensor_list.append(cur_combined_tensor)
                combined_mask_list.append(cur_mask_tensor)
                labels_tensor_list.append(cur_labels_tensor)
                nli_tensor_list.append(cur_nli_tensor)
                doc_mask_list.append(cur_doc_mask)

            combined_tensor = torch.stack(combined_tensor_list, dim=0)    # [B, num_ctx, L]
            combined_mask = torch.stack(combined_mask_list, dim=0)        # [B, num_ctx, L]
            labels_tensor = torch.stack(labels_tensor_list, dim=0)        # [B, num_ctx]
            nli_tensor = torch.stack(nli_tensor_list, dim=0)              # [B, num_ctx]
            doc_mask = torch.stack(doc_mask_list, dim=0)                  # [B, num_ctx]

            return {
                "combined": combined_tensor,
                "combined_mask": combined_mask,
                "labels": labels_tensor,
                "nli_scores": nli_tensor,
                "doc_mask": doc_mask,
                "sample_ids": sample_ids
            }

        return collate_fn