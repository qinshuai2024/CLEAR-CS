import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import logging
import sys
sys.path.append('..')
sys.path.append('.')
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
import time
import copy
import pickle
import random
import numpy as np
import csv
import argparse
import toml
import os
import datetime
import shutil

from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaConfig, RobertaTokenizer
from model.ance import ANCE
# from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
import transformers
from transformers import (
    BertModel,
    BertConfig,
    AutoTokenizer,
)
from utils.utils import check_dir_exist_or_build, pstore, pload, set_seed, get_optimizer, print_res
from dataset.qrecc import QReccDataset
from dataset.topiocqa import TopiocqaDataset


def cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs, device):
    batch_size = len(query_embs)
    pos_scores = query_embs.mm(pos_doc_embs.T)  # B * B
    neg_scores = torch.sum(query_embs * neg_doc_embs, dim = 1).unsqueeze(1) # B * 1 hard negatives
    score_mat = torch.cat([pos_scores, neg_scores], dim = 1)    # B * (B + 1)  in_batch negatives + 1 BM25 hard negative 
    label_mat = torch.arange(batch_size).to(device) # B
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(score_mat, label_mat)
    return loss

def cal_ranking_oracle_loss(query_embs, pos_doc_embs, device):
    batch_size = len(query_embs)
    pos_scores = query_embs.mm(pos_doc_embs.T)  # B * B
    score_mat = pos_scores
    #neg_scores = torch.sum(query_embs * neg_doc_embs, dim = 1).unsqueeze(1) # B * 1 hard negatives
    #score_mat = torch.cat([pos_scores, neg_scores], dim = 1)    # B * (B + 1)  in_batch negatives + 1 BM25 hard negative 
    label_mat = torch.arange(batch_size).to(device) # B
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(score_mat, label_mat)
    return loss

def cal_kd_loss(query_embs, kd_embs):
    loss_func = nn.MSELoss()
    return loss_func(query_embs, kd_embs)

def cal_mse_loss_terms(query_embs, doc_embs):
    batch_size = query_embs.size(0)
    embedding_dim = query_embs.size(1)

    mse_loss_func = nn.MSELoss()
    mse_loss = mse_loss_func(query_embs, doc_embs)

    query_embs_l2_norm = torch.linalg.norm(query_embs, ord=2, dim=1)
    doc_embs_l2_norm = torch.linalg.norm(doc_embs, ord=2, dim=1)
    norm_squared_sum = torch.square(query_embs_l2_norm) + torch.square(doc_embs_l2_norm)
    regularization_term = torch.mean(norm_squared_sum / embedding_dim, dim=0)

    dot_product = torch.einsum('ij, ij -> i', query_embs, doc_embs)
    negative_dot_product_term = torch.mean(- 2 / embedding_dim * dot_product, dim=0)
    #assert mse_loss == regularization_term + negative_dot_product_term
    return regularization_term, negative_dot_product_term


def save_model(args, model, tokenizer, save_type, epoch, step, loss):
    """Save model in pretrained format"""
    output_dir = os.path.join(
        args.model_output_path,
        f"{save_type}-model-epoch{epoch}-step{step}-loss{loss:.4f}"
    )
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle distributed/DP model
        model_to_save = model.module if hasattr(model, 'module') else model
        
        # Save model and tokenizer
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training metadata
        metadata = {
            'epoch': epoch,
            'step': step,
            'loss': float(loss),
            'save_type': save_type,
            'timestamp': datetime.datetime.now().isoformat(),
            'config': vars(args)
        }
        with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Saved {save_type} model to {output_dir}")
    except Exception as e:
        logging.error(f"Failed to save model: {str(e)}")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def train(args, local_rank, world_size=1):
    """Complete training function with model saving and resuming capability"""
    logging.info("Training parameters %s", args)
    if args.n_gpu > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",  # Use environment variables for automatic initialization
            rank=local_rank,
            world_size=world_size
        )
        logging.info(f"Initialized process group: rank {local_rank}, world_size {world_size}")
        torch.cuda.set_device(local_rank)
        
    config = RobertaConfig.from_pretrained(args.pretrained_query_encoder_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_query_encoder_path, do_lower_case=True)
    
    query_encoder = ANCE.from_pretrained(args.pretrained_query_encoder_path, config=config).to(local_rank)
    oracle_query_encoder = ANCE.from_pretrained(args.pretrained_oracle_encoder_path, config=config).to(local_rank)

    # DDP model wrapping
    if args.n_gpu > 1:
        query_encoder = DDP(query_encoder, device_ids=[local_rank], find_unused_parameters=True)
        oracle_query_encoder = DDP(oracle_query_encoder, device_ids=[local_rank], find_unused_parameters=True)
    
    args.batch_size = args.per_gpu_train_batch_size
    
    # data prepare
    if args.dataset == "topiocqa":
        train_dataset = TopiocqaDataset(args, tokenizer, args.train_file_path, args.rewrite_file_path, is_training=True)  #
    elif args.dataset == "qrecc":
        train_dataset = QReccDataset(args, tokenizer, args.train_file_path, is_training=True)
    # Single-machine multi-GPU data parallelism
    if args.n_gpu > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.n_gpu, rank=local_rank, shuffle=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,  
            collate_fn=train_dataset.get_collate_fn(args),
            sampler=train_sampler
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=train_dataset.get_collate_fn(args),
            shuffle=True
        )

    logging.info("train samples num = {}".format(len(train_dataset)))
    
    total_training_steps = args.num_train_epochs * (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))
    num_warmup_steps = args.num_warmup_portion * total_training_steps
    
    optimizer = get_optimizer(args, query_encoder, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)
    
    global_step = 0
    save_model_order = 0

    # begin to train
    logging.info("Start training...")
    logging.info("Total training epochs = {}".format(args.num_train_epochs))
    logging.info("Total training steps = {}".format(total_training_steps))
    
    num_steps_per_epoch = total_training_steps // args.num_train_epochs
    logging.info("Num steps per epoch = {}".format(num_steps_per_epoch))

    if isinstance(args.print_steps, float):
        args.print_steps = int(args.print_steps * num_steps_per_epoch)
        args.print_steps = max(1, args.print_steps)
    logging.info(f"print_steps: {args.print_steps}")
    start_epoch = 0
    epoch_iterator = trange(start_epoch, args.num_train_epochs, desc="Epoch", disable=args.disable_tqdm)

    best_loss = 1000
    for epoch in epoch_iterator:
        query_encoder.train()
        oracle_query_encoder.eval()
        # DDP sets sampling seed for each epoch to ensure different sampling
        if args.n_gpu > 1:
            train_dataloader.sampler.set_epoch(epoch)
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}", leave=True): 
            query_encoder.zero_grad()
            bt_conv_query = batch['bt_conv_qa'].to(local_rank) # B * len
            bt_conv_query_mask = batch['bt_conv_qa_mask'].to(local_rank)
            bt_oracle_query = batch['bt_oracle_utt'].to(local_rank)
            bt_oracle_query_mask = batch['bt_oracle_utt_mask'].to(local_rank)
            bt_pos_docs = batch['bt_pos_docs'].to(local_rank) # B * len one pos
            bt_pos_docs_mask = batch['bt_pos_docs_mask'].to(local_rank)
            bt_neg_docs = batch['bt_neg_docs'].to(local_rank) # B * len one pos
            bt_neg_docs_mask = batch['bt_neg_docs_mask'].to(local_rank)
            
            conv_query_embs = query_encoder(bt_conv_query, bt_conv_query_mask)  # B * dim
            with torch.no_grad():
                # freeze oracle query encoder's parameters
                if args.dense:
                    pos_doc_embs = oracle_query_encoder(bt_pos_docs, bt_pos_docs_mask).detach()  # B * dim
                    neg_doc_embs = oracle_query_encoder(bt_neg_docs, bt_neg_docs_mask).detach()  # B * dim
                else:
                    oracle_utt_embs = oracle_query_encoder(bt_oracle_query, bt_oracle_query_mask).detach()  # B * dim

            if args.dense:
                ranking_loss = cal_ranking_loss(conv_query_embs, pos_doc_embs, neg_doc_embs, device=local_rank)
                # Use original ANCE fine-tuning, only cross-entropy/contrastive learning
                loss = ranking_loss
            elif args.conv2query:    
                # Train query rewrite alignment
                oracle_loss = cal_kd_loss(conv_query_embs, oracle_utt_embs)
                loss = oracle_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()


            if args.print_steps > 0 and global_step % args.print_steps == 0: # oracle loss = {},  infusion loss = {}, ranking loss = {},
                logging.info("Epoch = {}, Global Step = {}, total loss = {}".format(
                                epoch + 1,
                                global_step,
                                loss.item(),
                                ))
                # regularization_term.item(),
                # ranking_loss.item(),
                # infusion_loss.item(),

            global_step += 1    # avoid saving the model of the first step.
            # save model finally
        # if best_loss > loss:
        if args.n_gpu == 1 or (args.n_gpu > 1 and local_rank == 0):
            save_model(args, query_encoder, tokenizer, save_model_order, epoch, global_step, loss.item())
            best_loss = loss.item()
            best_epoch = epoch
            logging.info("Epoch = {}, Global Step = {}, total loss = {}".format(
                            epoch + 1,
                            global_step,
                            loss.item()))
          
    # logging.info(f"Training completed. Best loss: {best_loss:.4f} at epoch {best_epoch}")
    if args.n_gpu > 1:
        # DDP cleanup processes
        dist.destroy_process_group()           

def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_query_encoder_path", type=str, default="checkpoint/ad-hoc-ance-msmarco")
    parser.add_argument("--pretrained_oracle_encoder_path", type=str, default="checkpoint/ad-hoc-ance-msmarco")

    parser.add_argument("--train_file_path", type=str, default="datasets/topiocqa/train_with_info.json")
    parser.add_argument("--rewrite_file_path", type=str, default="datasets/topiocqa/QR/train_T5QR.json")
    parser.add_argument('--model_output_path', type=str, default="output/topiocqa/model")
    parser.add_argument('--log_dir_path', type=str, default="loss_log")
    parser.add_argument("--collate_fn_type", type=str, default="flat_concat_for_train")
    parser.add_argument("--reranker_encoder_path", type=str, default="deepset/roberta-base-squad2")

    parser.add_argument("--per_gpu_train_batch_size", type=int, default=32)
    parser.add_argument("--use_data_percent", type=float, default=1)
    
    parser.add_argument("--num_train_epochs", type=int, default=20, help="num_train_epochs")
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length, consistent with \"Dialog inpainter\".")
    parser.add_argument("--max_response_length", type=int, default=64, help="Max response length, 64 for qrecc, 350 for cast20 since we only have one (last) response")
    parser.add_argument("--max_concat_length", type=int, default=512, help="Max concatenation length of the session. 512 for QReCC.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--disable_tqdm", type=bool, default=True)
    parser.add_argument("--mode", type=str, default="mse+CL")
    parser.add_argument("--dataset", type=str, default="topiocqa")
 
    parser.add_argument("--print_steps", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--num_warmup_portion", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--device", type=int, default=0)
    # parser.add_argument("--device2", type=int, default=0)
    
    parser.add_argument("--log_path", type=str)
    parser.add_argument("--sub_passge_embeddings_dir", type=str)
    
    parser.add_argument("--iteration_step", default=80, type=int)
    parser.add_argument("--iteration_reranker_step", default=40, type=int)
    parser.add_argument("--adv_lambda", default=0.6, type=float )
    parser.add_argument("--lambda_penalty", default=0.3, type=float)
    parser.add_argument("--logging_steps", default=40, type=int )
    parser.add_argument("--save_steps", default=1840, type=int )
    parser.add_argument("--temperature_normal", default=1.3, type=float )
    parser.add_argument("--resume_from_checkpoint", type=bool, default=False) 
    parser.add_argument("--num_hard_negatives", default=5, type=int)
    parser.add_argument("--hard_neg_extra_weight", default=1.0, type=float)
    parser.add_argument(
        "--max_steps",
        default=30000,
        type=int,
        help="If > 0: set total number of training steps to perform",
    )
    
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--conv2query", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    # log_writer = SummaryWriter(log_dir = args.log_dir_path)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                        filename=args.log_path)
    logging.info(f"args.dense:{args.dense}")
    logging.info(f"args.conv2query:{args.conv2query}")
    # args.n_gpu = torch.cuda.device_count()
    print(f"args.n_gpu: {args.n_gpu} Starting training")
    if args.n_gpu == 1:
        # Single GPU
        # train_adv(args, local_rank=args.device)
        train(args, local_rank=args.device)
    else:
        # Multi-GPU
        print("Multi-GPU")
        # os.environ["MASTER_PORT"] = "29501"
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ['LOCAL_RANK'])
        # train_adv(args, local_rank=local_rank, world_size=world_size)
        train(args, local_rank=local_rank, world_size=world_size)
    # log_writer.close()