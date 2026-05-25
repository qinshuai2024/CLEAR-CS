import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any


def _as_list(x: Any) -> List[float]:
    if isinstance(x, list):
        return [float(v) for v in x]
    if x is None:
        return []
    return [float(x)]


def _pair_keys(record: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    pairs: Dict[str, Tuple[str, str]] = {}
    for k in record.keys():
        if not k.endswith("_entail"):
            continue
        prefix = k[: -len("_entail")]
        s_key = f"{prefix}_sims"
        if s_key in record:
            pairs[prefix] = (k, s_key)
    return pairs


def _flatten(lst_of_lsts: List[List[float]]) -> List[float]:
    out: List[float] = []
    for lst in lst_of_lsts:
        out.extend(lst)
    return out


def adjust_record(record: Dict[str, Any], lam1: float, lam2: float, inplace: bool) -> Tuple[Dict[str, Any], bool]:
    pairs = _pair_keys(record)
    if not pairs:
        return record, False

    ans = str(record.get("answer", "")).strip()
    if ans.lower() == "unanswerable":
        modified = False
        for p in pairs.keys():
            e_key, s_key = pairs[p]
            s_vals = [min(v, 1.0) for v in _as_list(record.get(s_key, []))]
            if inplace:
                record[e_key] = s_vals
            else:
                record[f"{e_key}_adj"] = s_vals
            modified = True
        return record, modified

    pos_prefixes = [p for p in pairs.keys() if p.startswith("pos")]
    neg_prefixes = [p for p in pairs.keys() if p not in pos_prefixes]
    if not pos_prefixes or not neg_prefixes:
        return record, False

    pos_entails = _flatten([_as_list(record[pairs[p][0]]) for p in pos_prefixes])
    neg_entails = _flatten([_as_list(record[pairs[p][0]]) for p in neg_prefixes])
    if not pos_entails or not neg_entails:
        return record, False

    if not (max(pos_entails) < max(neg_entails)):
        return record, False

    neg_new_values: Dict[str, List[float]] = {}
    for p in neg_prefixes:
        e_key, s_key = pairs[p]
        e_vals = _as_list(record.get(e_key, []))
        s_vals = _as_list(record.get(s_key, []))
        n = min(len(e_vals), len(s_vals))
        new_vals = [min(e_vals[i] * (1.0 - lam2 * (1.0 - s_vals[i])), 1.0) for i in range(n)]
        if len(e_vals) > n:
            new_vals.extend([min(v, 1.0) for v in e_vals[n:]])
        neg_new_values[p] = new_vals

    pos_new_values: Dict[str, List[float]] = {}
    for p in pos_prefixes:
        e_key, s_key = pairs[p]
        e_vals = _as_list(record.get(e_key, []))
        s_vals = _as_list(record.get(s_key, []))
        n = min(len(e_vals), len(s_vals))
        new_vals = [min(e_vals[i] * (1.0 + lam1 * s_vals[i]), 1.0) for i in range(n)]
        if len(e_vals) > n:
            new_vals.extend([min(v, 1.0) for v in e_vals[n:]])
        pos_new_values[p] = new_vals

    modified = False
    for p, new_vals in neg_new_values.items():
        e_key, _ = pairs[p]
        if inplace:
            record[e_key] = new_vals
        else:
            record[f"{e_key}_adj"] = new_vals
        modified = True

    for p, new_vals in pos_new_values.items():
        e_key, _ = pairs[p]
        if inplace:
            record[e_key] = new_vals
        else:
            record[f"{e_key}_adj"] = new_vals
        modified = True

    return record, modified


def process_file(in_path: Path, out_path: Path, lam1: float, lam2: float, inplace: bool) -> Tuple[int, int]:
    total = 0
    changed = 0
    with in_path.open("r", encoding="utf-8") as fin, (
        sys.stdout if out_path is None else out_path.open("w", encoding="utf-8")
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line + "\n")
                continue
            obj2, mod = adjust_record(obj, lam1, lam2, inplace)
            if mod:
                changed += 1
            fout.write(json.dumps(obj2, ensure_ascii=False) + "\n")
    return total, changed


def main():
    parser = argparse.ArgumentParser(prog="reweight_entail_v11")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--lambda1", type=float, default=0.5)
    parser.add_argument("--lambda2", type=float, default=0.5)
    parser.add_argument("--inplace", action="store_true")
    args = parser.parse_args()

    if args.lambda1 < 0.0  or args.lambda2 < 0.0 or args.lambda2 > 1.0:
        print("lambda1 and lambda2 must be in [0,1]", file=sys.stderr)
        sys.exit(2)

    total, changed = process_file(args.input, args.output, args.lambda1, args.lambda2, args.inplace)
    print(f"Processed {total} lines, modified {changed} lines.", file=sys.stderr)


if __name__ == "__main__":
    main()


"""
python src/data_process/process_nli/reweight_entail.py \
  data/qrecc/new_data/nli/train_with_negs_entail_minmaxdot.json \
  -o data/qrecc/new_data/nli/train_with_negs_entail_minmaxdot_corr_11_08.json \
  --lambda1 1.1 --lambda2 0.8 --inplace

Processed 29596 lines, modified 5923 lines.

python src/data_process/process_nli/reweight_entail.py \
  data/topiocqa/merge_entail/train_bm25_hard_negs_topi_new_entail_minmaxdot.json \
  -o data/topiocqa/merge_entail/train_bm25_hard_negs_topi_new_entail_minmaxdot_4_9.json \
  --lambda1 4 --lambda2 0.9 --inplace
"""