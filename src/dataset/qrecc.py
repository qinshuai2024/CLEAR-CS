import json
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from dataset.data import padding_seq_to_same_length


class QReccDataset(Dataset):
    def __init__(self, args, tokenizer, filename, is_training=False):
        self.args = args
        self.tokenizer = tokenizer
        self.is_training = is_training

        with open(filename, encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        # n = int(args.use_data_percent * len(data))
        # if n < len(data):
        #     random.seed(args.seed)
        #     data = random.sample(data, n)

        self.data = []
        for rec in tqdm(data):
            pos_list = rec.get('pos_docs_pids', rec.get('pos_docs', []))
            if len(pos_list) > 0:
                self.data.append(rec)

    def __len__(self):
        return len(self.data)

    def _build_concat_query(self, record):
        args = self.args
        tok = self.tokenizer
        cur_utt_text = record['cur_utt_text']
        ctx_utts_text = record.get('ctx_utts_text', [])

        cur_ids = tok.encode(cur_utt_text, add_special_tokens=True,
                              max_length=args.max_query_length, truncation=True)
        flat_concat = list(cur_ids)

        for j in range(len(ctx_utts_text) - 1, -1, -1):
            max_len = args.max_response_length if (j % 2 == 1) else args.max_query_length
            utt = tok.encode(ctx_utts_text[j], add_special_tokens=True,
                             max_length=max_len, truncation=True)
            if len(flat_concat) + len(utt) > args.max_concat_length:
                remain = args.max_concat_length - len(flat_concat) - 1
                if remain > 0:
                    flat_concat += utt[:remain] + [utt[-1]]
                break
            else:
                flat_concat.extend(utt)
        return padding_seq_to_same_length(flat_concat, max_pad_length=args.max_concat_length)

    def _pick_negative(self, record):
        # Only use bm25 or random samples
        bm25_pool = record.get('bm25_rel_neg_text', record.get('bm25_hard_neg_text', []))
        random_pool = record.get('random_neg_text', record.get('irrel_neg_text', []))
        
        # Prefer bm25, otherwise use random
        if len(bm25_pool) > 0:
            return random.choice(bm25_pool)
        elif len(random_pool) > 0:
            return random.choice(random_pool)
        else:
            return None

    def __getitem__(self, idx):
        record = self.data[idx]
        args = self.args
        tok = self.tokenizer

        flat_concat, flat_concat_mask = self._build_concat_query(record)
        oracle_text = record.get('oracle_utt_text', record['cur_utt_text'])
        oracle_ids = tok.encode(oracle_text, add_special_tokens=True,
                                max_length=args.max_query_length, truncation=True)
        oracle_ids, oracle_mask = padding_seq_to_same_length(oracle_ids, max_pad_length=args.max_query_length)
        pos_list = record.get('pos_docs_text', record.get('pos_docs', []))
        pos_text = random.choice(pos_list)
        pos_ids = tok.encode(pos_text, add_special_tokens=True,
                                max_length=args.max_doc_length, truncation=True)
        pos_ids, pos_mask = padding_seq_to_same_length(pos_ids, max_pad_length=args.max_doc_length)
        sample = [
            record.get('sample_id', str(idx)),
            flat_concat, flat_concat_mask,
            pos_ids, pos_mask,
            oracle_ids, oracle_mask,
        ]
        if self.is_training:
            neg_text = self._pick_negative(record)
            if neg_text is None:
                neg_text = pos_text
            neg_ids = tok.encode(neg_text, add_special_tokens=True,
                                 max_length=args.max_doc_length, truncation=True)
            neg_ids, neg_mask = padding_seq_to_same_length(neg_ids, max_pad_length=args.max_doc_length)
            sample.append(neg_ids)
            sample.append(neg_mask)
        return sample

    @staticmethod
    def get_collate_fn(args):
        def collate_fn(batch):
            collated = {
                'bt_sample_ids': [],
                'bt_conv_qa': [],
                'bt_conv_qa_mask': [],
                'bt_pos_docs': [],
                'bt_pos_docs_mask': [],
                'bt_oracle_utt': [],
                'bt_oracle_utt_mask': [],
            }
            if len(batch[0]) > 8:
                collated['bt_neg_docs'] = []
                collated['bt_neg_docs_mask'] = []
            for ex in batch:
                collated['bt_sample_ids'].append(ex[0])
                collated['bt_conv_qa'].append(ex[1])
                collated['bt_conv_qa_mask'].append(ex[2])
                collated['bt_pos_docs'].append(ex[3])
                collated['bt_pos_docs_mask'].append(ex[4])
                collated['bt_oracle_utt'].append(ex[5])
                collated['bt_oracle_utt_mask'].append(ex[6])
                if len(ex) > 7:
                    collated['bt_neg_docs'].append(ex[7])
                    collated['bt_neg_docs_mask'].append(ex[8])
            not_tensor = {'bt_sample_ids'}
            for k in collated:
                if k not in not_tensor:
                    collated[k] = torch.tensor(collated[k], dtype=torch.long)
            return collated
        return collate_fn

