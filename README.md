# From Topical Relevance to Answerability: Entailment Distillation for Conversational Retrieval

This repository contains the code for reproducing the retrieval and reranking experiments in the submitted paper. The implementation includes conversational dense retrieval, abductive query retrieval, NLI-based label construction, and answerability-aware reranking.

## Environment

The code was developed with Python 3.10.

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version before running GPU experiments.

## Repository Structure

```text
scripts/                         Running examples for retrieval and reranking
src/retrieval/                   Dense retrieval and abductive query retrieval
src/rerank/                      Reranker training and evaluation
src/data_process/preprocessing/  Dataset preprocessing utilities
src/data_process/index/          Dense index construction utilities
src/data_process/process_nli/    NLI-label generation and calibration
src/dataset/                     Dataset loaders
src/model/                       Retrieval and reranking models
```

## Data

Download the original datasets from their official sources:

- QReCC: https://github.com/apple/ml-qrecc
- TopiOCQA: https://github.com/McGill-NLP/topiocqa
- TREC CAsT: https://www.treccast.ai/

The scripts assume the following local layout by default:

```text
data/
  qrecc/
  topiocqa/
  cast/
experiments/
```

Large processed files, dense indexes, generated abductive queries, and model checkpoints are not included in this repository. Update the paths in `scripts/*.sh` to match your local data and checkpoint locations.

## Preprocessing

Dataset preprocessing utilities are under:

```bash
src/data_process/preprocessing
```

Dense index construction utilities are under:

```bash
src/data_process/index
```

NLI teacher scores and calibrated entailment labels can be generated with:

```bash
python src/data_process/process_nli/preprocess_nli_labels.py
python src/data_process/process_nli/compute_ance_sims.py
python src/data_process/process_nli/reweight_entail.py
```

## Retrieval

Run conversational dense retrieval:

```bash
bash scripts/test_base.sh
```

Run abductive query retrieval:

```bash
bash scripts/test_conv2query.sh
```

The abductive query generation pipeline is implemented in:

```bash
src/retrieval/finetune_llama3_a6000.py
src/retrieval/llama3_infer_a6000.py
```

LLM fine-tuning and query generation are offline preprocessing steps. They are not part of online inference.

## Reranking

Train the reranker:

```bash
bash scripts/train_reranker.sh
```

Evaluate the reranker:

```bash
bash scripts/test_reranker.sh
```

The main reranking modes are:

- `cls`: contextual relevance score only
- `nli`: entailment score only
- `all`: combined relevance and entailment score

## Notes

- The shell scripts are examples and should be edited to point to your local dataset, index, and checkpoint paths.
- For CAsT zero-shot evaluation, use parameters selected on the source dataset validation split rather than tuning on CAsT qrels.
- Generated files, checkpoints, logs, and experiment outputs should remain under `data/` or `experiments/`, which are ignored by default.
