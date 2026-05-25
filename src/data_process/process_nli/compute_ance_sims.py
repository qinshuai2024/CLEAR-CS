import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Optional
from collections import OrderedDict
import orjson

import torch
import torch.nn.functional as F
from tqdm import tqdm

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from model.ance import load_model  # noqa: E402


def build_model(model_path: str, device: str):
    tokenizer, model = load_model("ANCE_Passage", model_path)
    model.eval()
    model.to(device)
    if device.startswith("cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            model.half()
            model = torch.compile(model)
        except Exception:
            pass
    return tokenizer, model


def encode_texts(texts: List[str], tokenizer, model, device: str, batch_size: int = 16) -> torch.Tensor:
    embs = []
    if len(texts) == 0:
        hidden = getattr(model, "embeddingHead").out_features if hasattr(model, "embeddingHead") else 768
        return torch.empty((0, hidden), device=device)
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            doc_vecs = model.doc_emb(input_ids=input_ids, attention_mask=attention_mask)
            embs.append(doc_vecs)
    return torch.cat(embs, dim=0)


class LRUCache:
    def __init__(self, max_items: int = 100000):
        self.max_items = max_items
        self._map = OrderedDict()

    def get(self, key: str):
        v = self._map.get(key)
        if v is not None:
            self._map.move_to_end(key)
        return v

    def put(self, key: str, value: torch.Tensor):
        self._map[key] = value
        self._map.move_to_end(key)
        if len(self._map) > self.max_items:
            self._map.popitem(last=False)


def get_embeddings_with_cache(texts: List[str], tokenizer, model, device: str, batch_size: int, cache: Optional[LRUCache]) -> torch.Tensor:
    if len(texts) == 0:
        hidden = getattr(model, "embeddingHead").out_features if hasattr(model, "embeddingHead") else 768
        return torch.empty((0, hidden), device=device)
    embs_cpu: List[torch.Tensor] = [None] * len(texts)  # type: ignore
    missing_texts: List[str] = []
    missing_idx: List[int] = []
    if cache is not None:
        for i, t in enumerate(texts):
            v = cache.get(t)
            if v is None:
                missing_texts.append(t)
                missing_idx.append(i)
            else:
                embs_cpu[i] = v
    else:
        missing_texts = texts
        missing_idx = list(range(len(texts)))

    if missing_texts:
        new_embs_dev = encode_texts(missing_texts, tokenizer, model, device, batch_size=batch_size)
        new_embs_cpu = new_embs_dev.detach().to("cpu")
        for j, e in zip(missing_idx, new_embs_cpu):
            if cache is not None:
                cache.put(texts[j], e)
            embs_cpu[j] = e

    stacked = torch.stack(embs_cpu, dim=0)  # type: ignore
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    return stacked.to(device=device, dtype=dtype)


def detect_input_format(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return "jsonl"
            if ch.isspace():
                continue
            if ch == "[":
                return "json"
            return "jsonl"


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield orjson.loads(s)


def load_json_array(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_sims_for_sample(sample: Dict, tokenizer, model, device: str, batch_size: int, cache: Optional[LRUCache]) -> Dict:
    pos_docs = sample.get("pos_docs") or sample.get("pos_docs_text") or []
    if not pos_docs:
        return {"sample_id": sample.get("sample_id"), "error": "no_pos_doc"}

    result = dict(sample)

    neg_cat_keys: List[str] = []
    cat_to_indices: Dict[str, List[int]] = {}
    all_neg_texts: List[str] = []
    for key, val in list(sample.items()):
        if key.endswith("_neg_text") and isinstance(val, list):
            neg_cat_keys.append(key)
            start = len(all_neg_texts)
            all_neg_texts.extend(val)
            cat_to_indices[key] = list(range(start, start + len(val)))

    docs_texts = pos_docs + all_neg_texts
    P = len(pos_docs)
    if len(docs_texts) == 0:
        return {"sample_id": sample.get("sample_id"), "error": "no_docs"}

    uniq_texts: List[str] = []
    idx_map: List[int] = []
    seen: Dict[str, int] = {}
    for t in docs_texts:
        if t in seen:
            idx_map.append(seen[t])
        else:
            seen[t] = len(uniq_texts)
            idx_map.append(seen[t])
            uniq_texts.append(t)

    uniq_embs = get_embeddings_with_cache(uniq_texts, tokenizer, model, device, batch_size=batch_size, cache=cache)
    idx_tensor = torch.tensor(idx_map, device=uniq_embs.device, dtype=torch.long)
    pos_idx_tensor = torch.tensor(idx_map[:P], device=uniq_embs.device, dtype=torch.long)
    pos_embs = uniq_embs.index_select(0, pos_idx_tensor)

    Suniq = torch.matmul(pos_embs, uniq_embs.T)
    s_min = torch.min(Suniq)
    s_max = torch.max(Suniq)
    denom = (s_max - s_min).clamp(min=1e-12)
    Suniq_norm = (Suniq - s_min) / denom
    doc_scores_uniq = Suniq_norm.max(dim=0).values
    doc_scores = doc_scores_uniq.index_select(0, idx_tensor)

    pos_sims = [float(doc_scores[i].item()) for i in range(P)]
    result["pos_docs_sims"] = pos_sims

    offset = P
    for key in neg_cat_keys:
        idxs = cat_to_indices[key]
        sims = [float(doc_scores[offset + i].item()) for i in idxs]
        out_key = key.replace("neg_text", "neg_sims")
        result[out_key] = sims

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--model_path",
        default="3ricL/ad-hoc-ance-msmarco",
        help="Path to ANCE model or HuggingFace model ID"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cache_size", type=int, default=100000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_format", choices=["json", "jsonl"], default="json")
    parser.add_argument("--sim", choices=["minmax_dot"], default="minmax_dot")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tokenizer, model = build_model(args.model_path, args.device)

    fmt = detect_input_format(args.input_path)

    results: List[Dict] = []
    if fmt == "json":
        samples = load_json_array(args.input_path)
        cache = LRUCache(args.cache_size) if args.cache_size and args.cache_size > 0 else None
        for sample in tqdm(samples, desc="Computing ANCE sims"):
            r = compute_sims_for_sample(sample, tokenizer, model, args.device, args.batch_size, cache)
            results.append(r)
    else:
        if args.output_format == "jsonl":
            with open(args.output_path, "w", encoding="utf-8") as outf:
                cache = LRUCache(args.cache_size) if args.cache_size and args.cache_size > 0 else None
                for sample in tqdm(iter_jsonl(args.input_path), desc="Computing ANCE sims"):
                    r = compute_sims_for_sample(sample, tokenizer, model, args.device, args.batch_size, cache)
                    outf.write(orjson.dumps(r).decode("utf-8") + "\n")
            return
        else:
            cache = LRUCache(args.cache_size) if args.cache_size and args.cache_size > 0 else None
            for sample in tqdm(iter_jsonl(args.input_path), desc="Computing ANCE sims"):
                r = compute_sims_for_sample(sample, tokenizer, model, args.device, args.batch_size, cache)
                results.append(r)

    if args.output_format == "jsonl":
        with open(args.output_path, "w", encoding="utf-8") as outf:
            for r in results:
                outf.write(orjson.dumps(r).decode("utf-8") + "\n")
    else:
        with open(args.output_path, "w", encoding="utf-8") as outf:
            json.dump(results, outf, ensure_ascii=False)


if __name__ == "__main__":
    main()


"""
Example usage:

python src/data_process/process_nli/compute_ance_sims.py \
  --input_path data/input.jsonl \
  --output_path data/output.jsonl \
  --output_format jsonl \
  --batch_size 64 \
  --cache_size 200000 \
  --model_path 3ricL/ad-hoc-ance-msmarco \
  --device cuda
  
CUDA_VISIBLE_DEVICES=1 python src/data_process/process_nli/compute_ance_sims.py \
  --input_path data/qrecc/nli/train_with_negs_entail.json \
  --output_path data/qrecc/nli/train_with_negs_entail_minmaxdot.json \
  --output_format jsonl \
  --batch_size 64 \
  --cache_size 200000 \
  --model_path 3ricL/ad-hoc-ance-msmarco

"""