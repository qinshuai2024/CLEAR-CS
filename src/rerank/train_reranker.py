import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import logging
import sys
sys.path.append('..')
sys.path.append('.')
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)gen
import argparse
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

from model.reranker import QANLIReranker
from utils.utils import set_seed, get_optimizer
from dataset.reranker import RerankerQANLIDataset


def train_qanli_reranker(args, local_rank, world_size=1):
    logging.info("Training parameters %s", args)

    global_step = 0
    best_loss = float("inf")

    # DDP init
    if args.n_gpu > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=local_rank,
            world_size=world_size
        )
        logging.info(f"Initialized process group: rank {local_rank}, world_size {world_size}")
        torch.cuda.set_device(local_rank)

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.reranker_encoder_path)
    model = QANLIReranker(model_name=args.reranker_encoder_path).to(local_rank)

    if args.n_gpu > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = get_optimizer(args, model, weight_decay=args.weight_decay)

    # dataset
    train_dataset = RerankerQANLIDataset(
        file_path=args.train_file_path,
        reranker_tokenizer=tokenizer,
        num_hard_negatives=args.num_hard_negatives,
        max_query_length=args.max_query_length,
        max_response_length=args.max_response_length,
        max_concat_length=args.max_concat_length,
        max_doc_length=args.max_doc_length,
        include_context=True,
        is_training=True,
        dataset=args.dataset,
        bm25_per_sample=args.bm25_per_sample,
        topic_per_sample=args.topic_per_sample,
        random_per_sample=args.random_per_sample,
    )

    if args.n_gpu > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.n_gpu, rank=local_rank, shuffle=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            collate_fn=train_dataset.get_collate_fn(pad_token_id=tokenizer.pad_token_id, global_max_length=512),
            sampler=train_sampler
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            collate_fn=train_dataset.get_collate_fn(pad_token_id=tokenizer.pad_token_id, global_max_length=512),
            shuffle=True
        )

    max_step = args.max_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * max_step),
        num_training_steps=int(max_step)
    )

    for epoch in range(args.num_train_epochs):
        if args.n_gpu > 1:
            train_dataloader.sampler.set_epoch(epoch)
        if args.n_gpu > 1:
            model.module.train()
        else:
            model.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}", leave=True):
            input_ids = batch["combined"].to(local_rank)          # [B, M, L]
            attention_mask = batch["combined_mask"].to(local_rank) # [B, M, L]
            nli_teacher = batch["nli_scores"].to(local_rank)       # [B, M]
            doc_mask = batch.get("doc_mask").to(local_rank).float() # [B, M]

            # forward
            relevance_scores, entail_probs = model(input_ids, attention_mask)  # [B, M], [B, M]

            # binary classification: first doc is positive
            B, M = relevance_scores.shape
            binary_targets = torch.zeros_like(relevance_scores)
            binary_targets[:, 0] = 1.0

            bce_logits = F.binary_cross_entropy_with_logits(
                relevance_scores, binary_targets, reduction="none"
            )  # [B, M]
            denom_cls = torch.clamp(doc_mask.sum(), min=1.0)
            loss_cls = (bce_logits * doc_mask).sum() / denom_cls

            # NLI distillation (prob-prob BCE)
            bce_nli = F.binary_cross_entropy(entail_probs, nli_teacher, reduction="none")  # [B, M]
            denom_nli = torch.clamp(doc_mask.sum(), min=1.0)
            loss_nli = (bce_nli * doc_mask).sum() / denom_nli
            
            if args.mode == "cls":
                loss = loss_cls
            elif args.mode == "nli":
                loss = loss_nli
            elif args.mode == "all":
                loss = loss_cls + args.lambda_nli * loss_nli
            else:
                raise ValueError(f"Invalid mode, should be one of cls, nli, or all: {args.mode}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1
            num_batches += 1

            if global_step % args.logging_steps == 0:
                logging.info(f"Step {global_step}, batch loss: {loss.item():.4f}, cls: {loss_cls.item():.4f}, nli: {loss_nli.item():.4f}")

        avg_epoch_loss = epoch_loss / max(1, num_batches)
        logging.info(f"Epoch {epoch + 1} finished, average loss: {avg_epoch_loss:.4f}")

        # save best
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_path = f"{args.output_dir}/best_reranker_model_{epoch}.pt"
            if args.n_gpu > 1:
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            logging.info(f"New best model saved at {save_path} with loss {best_loss:.4f}")

    if args.n_gpu > 1:
        dist.destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--reranker_encoder_path", type=str, default="deepset/roberta-base-squad2")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--train_file_path", type=str, default="")
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument('--output_dir', type=str, default="./output_qanli")

    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_doc_length", type=int, default=384)
    parser.add_argument("--max_response_length", type=int, default=64)
    parser.add_argument("--max_concat_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dataset", type=str, default="topiocqa")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lambda_nli", type=float, default=0.5)

    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n_gpu", type=int, default=torch.cuda.device_count())

    parser.add_argument("--log_path", type=str, default="./log_qanli_reranker.txt")

    parser.add_argument("--logging_steps", default=200, type=int)

    # per-type negatives
    parser.add_argument("--num_hard_negatives", default=2, type=int, help="count for high_rel negatives")
    parser.add_argument("--bm25_per_sample", default=2, type=int)
    parser.add_argument("--topic_per_sample", default=2, type=int)
    parser.add_argument("--random_per_sample", default=2, type=int)

    parser.add_argument("--max_steps", default=30000, type=int)
    parser.add_argument("--mode", default="all", type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                        filename=args.log_path)

    print(f"args.n_gpu: {args.n_gpu} Starting training")
    if args.n_gpu == 1:
        train_qanli_reranker(args, local_rank=args.device)
    else:
        world_size = int(os.environ["WORLD_SIZE"]) \
            if "WORLD_SIZE" in os.environ else args.n_gpu
        local_rank = int(os.environ['LOCAL_RANK']) \
            if 'LOCAL_RANK' in os.environ else 0
        train_qanli_reranker(args, local_rank=local_rank, world_size=world_size)
