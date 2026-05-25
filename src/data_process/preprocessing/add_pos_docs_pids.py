import csv
import json
import os


def read_qrel(qrel_path: str, min_rel: int = 1):
    qid_to_docids = {}
    with open(qrel_path, "r", encoding="utf-8") as f:
        tsv_reader = csv.reader(f, delimiter="\t")
        for row in tsv_reader:
            if not row or len(row) < 4:
                continue
            qid = str(row[0])
            docid = str(row[2])
            try:
                rel = int(row[3])
            except ValueError:
                continue
            if rel >= min_rel:
                qid_to_docids.setdefault(qid, [])
                if docid not in qid_to_docids[qid]:
                    qid_to_docids[qid].append(docid)
    return qid_to_docids


def add_pos_docs_pids(jsonl_path: str, qrel_path: str, output_path: str | None = None, min_rel: int = 1) -> str:
    qid_to_docids = read_qrel(qrel_path, min_rel=min_rel)
    if output_path is None:
        base, ext = os.path.splitext(jsonl_path)
        output_path = f"{base}_with_pos_pids.jsonl"
    with open(jsonl_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("sample_id", ""))
            obj["pos_docs_pids"] = [int(pid) for pid in qid_to_docids.get(qid, [])]
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return output_path



# cast19_path = 'data/cast/data_cast19/new_data/cast19_test_topiocqa.jsonl'
# cast19_rel = 'data/cast/data_cast19/new_data/cast19_qrel.tsv'
# print(add_pos_docs_pids(cast19_path, cast19_rel))

cast20_path = 'data/cast/data_cast20/new_data/cast20_test_topiocqa.jsonl'
cast20_rel = 'data/cast/data_cast20/new_data/cast20_qrel.tsv'
print(add_pos_docs_pids(cast20_path, cast20_rel))
