import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import random

def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids
            
    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length
  
    return input_ids, attention_mask

class TestRerankQreccDataset(Dataset):
    def __init__(self, file_path, tokenizer,
                 max_query_length=32, max_response_length=64, max_concat_length=256,
                 max_doc_length=512, include_context=True, rank_k=10, dataset="qrecc"):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_response_length = max_response_length
        self.max_concat_length = max_concat_length
        self.max_doc_length = max_doc_length
        self.include_context = include_context
        self.topk = rank_k
        self.dataset = dataset
        self.data = self._load_data(file_path)

    def __len__(self):
        return len(self.data)

    def _load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        valid_data = []
        for line in tqdm(data):
            sample = json.loads(line.strip())
            # Ensure candidate passages exist
            # if "topk_text" in sample and len(sample["topk_text"]) == 10:
            valid_data.append(sample)
        print(f"Loaded {len(valid_data)} valid samples with {self.topk} docs each")
        return valid_data

    def _build_concat_query(self, sample):
        """Build concatenated query"""
        if self.dataset == "qrecc":
            cur_utt = sample['cur_utt_text'].strip()
            ctx_utts_text = sample['ctx_utts_text']
        else:
            ctx_utts_text = sample['cur_utt_text'].strip().split(" [SEP] ") # [q1, a1, q2, a2, ...]
            cur_utt = ctx_utts_text[-1] 
            ctx_utts_text = ctx_utts_text[:-1]
            
        cur_utt = self.tokenizer.encode(
            cur_utt,
            add_special_tokens=True,
            max_length=self.max_query_length,
            truncation=True
        )
        concat_query = cur_utt.copy()

        # if not self.include_context or 'ctx_utts_text' not in sample:
        #     return concat_query

        # Concatenate context in reverse order
        for j in range(len(ctx_utts_text) - 1, -1, -1):
            is_response = j % 2 == 1
            max_len = self.max_response_length if is_response else self.max_query_length

            utt = self.tokenizer.encode(
                ctx_utts_text[j].strip(),
                add_special_tokens=True,
                max_length=max_len,
                truncation=True
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
        return self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_doc_len,
            truncation=True
        )

    def __getitem__(self, index):
        sample = self.data[index]
        concat_query = self._build_concat_query(sample)
        # # Use answer
        # concat_query = self.tokenizer.encode(
        #     sample["answer"].strip(),
        #     add_special_tokens=True,
        #     max_length=self.max_response_length,
        #     truncation=True
        # )
        # # Use rewritten query for ranking, see the effect
        # rewritten_query = sample["oracle_utt_text"]
        # concat_query = self.tokenizer.encode(
        #     rewritten_query,
        #     add_special_tokens=True,
        #     max_length=self.max_query_length,
        #     truncation=True
        # )

        docs = sample['topk_text'][:self.topk]  
        doc_ids = sample["topk_id"][:self.topk]
        query_len = len(concat_query)
        # print(f"Total number of documents: {len(doc_ids)}")
        # Build query+doc sequence
        combined_ids = [
            concat_query + self._encode_text(doc, self.max_doc_length, query_len)[1:]  # Remove CLS token from doc
            for doc in docs
        ]

        return {
            "combined": combined_ids,  # list of 10
            "doc_ids": doc_ids,        # Corresponding passage ids
            "sample_id": sample.get('sample_id', str(index))
        }

    @staticmethod
    def get_collate_fn(pad_token_id=0, global_max_length=512):
        def collate_fn(batch):
            batch_size = len(batch)
            num_docs = len(batch[0]["combined"])  # 10
            # Pad each query+doc
            def pad_seq(seq):
                seq = seq[:global_max_length]
                return seq + [pad_token_id] * (global_max_length - len(seq))

            # Generate 3D tensor [batch, 10, seq_len]
            combined_padded = [
                [pad_seq(doc) for doc in item["combined"]] for item in batch
            ]
            combined_tensor = torch.LongTensor(combined_padded)
            combined_mask = (combined_tensor != pad_token_id).long()

            sample_ids = [item["sample_id"] for item in batch]
            doc_ids = [item["doc_ids"] for item in batch]

            return {
                "combined": combined_tensor,      # [B, 10, L]
                "combined_mask": combined_mask,   # [B, 10, L]
                "sample_ids": sample_ids,
                "doc_ids": doc_ids
            }
        return collate_fn

class TestNLIQreccDataset(Dataset):
    def __init__(self, file_path, tokenizer, rank_k=10, max_length=512, dataset="qrecc"):
        self.tokenizer = tokenizer
        self.topk = rank_k
        self.max_length = max_length
        self.dataset = dataset
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line.strip()) for line in f]
        print(f"Loaded {len(self.data)} valid samples with top docs each")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        query = sample["cur_utt_text"]
        ctx = sample.get("ctx_utts_text", [])
        if self.dataset == "qrecc":
            answer = sample["answer"]
        else:
            answer = sample["answer"]

        query_ctx = "\n".join(ctx) + "\n" + query
        hypothesis = query_ctx + "\n" + answer
        # hypothesis = answer

        premises = sample['topk_text'][:self.topk]
        doc_ids = sample["topk_id"][:self.topk]

        tokenized = [
            self.tokenizer(
                premise,
                hypothesis,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            for premise in premises
        ]
        # print("docis:", len(doc_ids))
        doc_ids = torch.tensor(doc_ids)
        return {
            "input_ids": torch.stack([t["input_ids"].squeeze(0) for t in tokenized]),        # [topk, L]
            "attention_mask": torch.stack([t["attention_mask"].squeeze(0) for t in tokenized]),  # [topk, L]
            "doc_ids": doc_ids,
            "sample_ids": sample.get("sample_id", str(index))
        }





