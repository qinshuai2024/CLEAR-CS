import json
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RerankerQANLIDataset(Dataset):
    def __init__(self, file_path, reranker_tokenizer,
                 num_hard_negatives=1,
                 max_query_length=32, max_response_length=64,
                 max_concat_length=256, max_doc_length=512,
                 include_context=True, is_training=True, dataset="topiocqa",
                 bm25_per_sample=2, topic_per_sample=3, random_per_sample=4):
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

        pos_list = sample.get('pos_docs_text', sample.get('pos_docs', []))
        pos_idx = random.sample(range(len(pos_list)), 1)[0]
        pos_doc = pos_list[pos_idx]
        pos_entails = sample.get('pos_docs_entail', [0.0] * len(pos_list))
        pos_score = float(pos_entails[pos_idx]) if pos_idx < len(pos_entails) else 0.0

        high_list = sample.get('hight_rel_neg_text', [])
        high_entails = sample.get('hight_rel_neg_entail', [0.0] * len(high_list))
        bm25_list = sample.get('bm25_rel_neg_text', [])
        bm25_entails = sample.get('bm25_rel_neg_entail', [0.0] * len(bm25_list))

        target_high = self.num_hard_negatives
        # within-class selection/replication; if empty, fill with placeholders
        if len(high_list) == 0:
            high_sel = [""] * target_high
            high_scores = [0.0] * target_high
        elif len(high_list) < target_high:
            repeats = (target_high + len(high_list) - 1) // len(high_list)
            idx = (list(range(len(high_list))) * repeats)[:target_high]
            high_sel = [high_list[i] for i in idx]
            high_scores = [float(high_entails[i]) if i < len(high_entails) else 0.0 for i in idx]
        else:
            sel_idx = random.sample(range(len(high_list)), target_high) if self.is_training else list(range(target_high))
            high_sel = [high_list[i] for i in sel_idx]
            high_scores = [float(high_entails[i]) if i < len(high_entails) else 0.0 for i in sel_idx]

        if len(bm25_list) == 0:
            bm25_sel = [""] * self.bm25_per_sample
            bm25_scores = [0.0] * self.bm25_per_sample
        elif len(bm25_list) < self.bm25_per_sample:
            repeats = (self.bm25_per_sample + len(bm25_list) - 1) // len(bm25_list)
            bm25_sel_idx = (list(range(len(bm25_list))) * repeats)[:self.bm25_per_sample]
            bm25_sel = [bm25_list[i] for i in bm25_sel_idx]
            bm25_scores = [float(bm25_entails[i]) if i < len(bm25_entails) else 0.0 for i in bm25_sel_idx]
        else:
            bm25_sel_idx = random.sample(range(len(bm25_list)), self.bm25_per_sample) if self.is_training else list(range(self.bm25_per_sample))
            bm25_sel = [bm25_list[i] for i in bm25_sel_idx]
            bm25_scores = [float(bm25_entails[i]) if i < len(bm25_entails) else 0.0 for i in bm25_sel_idx]

        topic_list = sample.get('topic_shift_neg_text', [])
        topic_entails = sample.get('topic_shift_neg_entail', [0.0] * len(topic_list))
        if len(topic_list) == 0:
            topic_sel = [""] * self.topic_per_sample
            topic_scores = [0.0] * self.topic_per_sample
        elif len(topic_list) < self.topic_per_sample:
            repeats = (self.topic_per_sample + len(topic_list) - 1) // len(topic_list)
            topic_sel_idx = (list(range(len(topic_list))) * repeats)[:self.topic_per_sample]
            topic_sel = [topic_list[i] for i in topic_sel_idx]
            topic_scores = [float(topic_entails[i]) if i < len(topic_entails) else 0.0 for i in topic_sel_idx]
        else:
            topic_sel_idx = random.sample(range(len(topic_list)), self.topic_per_sample) if self.is_training else list(range(self.topic_per_sample))
            topic_sel = [topic_list[i] for i in topic_sel_idx]
            topic_scores = [float(topic_entails[i]) if i < len(topic_entails) else 0.0 for i in topic_sel_idx]

        rand_list = sample.get('random_neg_text', [])
        rand_entails = sample.get('random_neg_entail', [0.0] * len(rand_list))
        if len(rand_list) == 0:
            rand_sel = [""] * self.random_per_sample
            rand_scores = [0.0] * self.random_per_sample
        elif len(rand_list) < self.random_per_sample:
            repeats = (self.random_per_sample + len(rand_list) - 1) // len(rand_list)
            rand_sel_idx = (list(range(len(rand_list))) * repeats)[:self.random_per_sample]
            rand_sel = [rand_list[i] for i in rand_sel_idx]
            rand_scores = [float(rand_entails[i]) if i < len(rand_entails) else 0.0 for i in rand_sel_idx]
        else:
            rand_sel_idx = random.sample(range(len(rand_list)), self.random_per_sample) if self.is_training else list(range(self.random_per_sample))
            rand_sel = [rand_list[i] for i in rand_sel_idx]
            rand_scores = [float(rand_entails[i]) if i < len(rand_entails) else 0.0 for i in rand_sel_idx]

        docs = [pos_doc] + high_sel + bm25_sel + topic_sel + rand_sel
        nli_scores = [pos_score] + high_scores + bm25_scores + topic_scores + rand_scores
        labels = [1] + [0] * (len(docs) - 1)

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
            lens = [len(it["combined"]) for it in batch]
            num_ctx = max(lens)

            def pad_ids(seq, max_len):
                seq = seq[:max_len]
                return seq + [pad_token_id] * (max_len - len(seq))

            combined_tensor_list = []
            combined_mask_list = []
            labels_tensor_list = []
            nli_tensor_list = []
            doc_mask_list = []

            for it in batch:
                cur_combined = [pad_ids(c, global_max_length) for c in it["combined"]]
                cur_labels = it["labels"]
                cur_nli = it["nli_scores"]

                cur_len = len(cur_combined)
                if cur_len < num_ctx:
                    pad_doc = [pad_token_id] * global_max_length
                    cur_combined += [pad_doc] * (num_ctx - cur_len)
                    cur_labels += [0] * (num_ctx - cur_len)
                    cur_nli += [0.0] * (num_ctx - cur_len)

                cur_combined_tensor = torch.LongTensor(cur_combined)
                cur_mask_tensor = (cur_combined_tensor != pad_token_id).long()
                cur_labels_tensor = torch.LongTensor(cur_labels)
                cur_nli_tensor = torch.FloatTensor(cur_nli)
                cur_doc_mask = torch.zeros(num_ctx, dtype=torch.long)
                cur_doc_mask[:len(it["combined"])] = 1

                combined_tensor_list.append(cur_combined_tensor)
                combined_mask_list.append(cur_mask_tensor)
                labels_tensor_list.append(cur_labels_tensor)
                nli_tensor_list.append(cur_nli_tensor)
                doc_mask_list.append(cur_doc_mask)

            combined_tensor = torch.stack(combined_tensor_list, dim=0)
            combined_mask = torch.stack(combined_mask_list, dim=0)
            labels_tensor = torch.stack(labels_tensor_list, dim=0)
            nli_tensor = torch.stack(nli_tensor_list, dim=0)
            doc_mask = torch.stack(doc_mask_list, dim=0)

            return {
                "combined": combined_tensor,
                "combined_mask": combined_mask,
                "labels": labels_tensor,
                "nli_scores": nli_tensor,
                "doc_mask": doc_mask,
                "sample_ids": sample_ids
            }
        return collate_fn
