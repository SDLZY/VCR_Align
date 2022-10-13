"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER finetuning for VCR
"""
import argparse
import json
import os
from os.path import exists, join
from time import time

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import Adam, Adamax

from apex import amp
from horovod import torch as hvd

from tqdm import tqdm

from data import (TokenBucketSampler, PrefetchLoader, DetectFeatLmdb,
                  VcrTxtTokLmdb, ImageLmdbGroup, ConcatDatasetWithLens,
                  VcrDataset, VcrEvalDataset,
                  vcr_collate, vcr_eval_collate,)
from model.vcr import UniterForVisualCommonsenseReasoning
from optim import AdamW, get_lr_sched

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.const import BUCKET_SIZE, IMG_DIM
from model.losses import get_align_model
import numpy as np

import cytoolz
import h5py
import copy
import shutil
import copy
NUM_SPECIAL_TOKENS = 81
torch.set_printoptions(threshold=np.inf, linewidth=1000000)
logitorprob='probs'

def build_dataloader(dataset, collate_fn, is_train, opts):
    batch_size = (opts.train_batch_size if is_train
                  else opts.val_batch_size)
    if is_train:
        sampler = TokenBucketSampler(dataset.lens, bucket_size=BUCKET_SIZE,
                                     batch_size=batch_size, droplast=is_train)
        dataloader = DataLoader(dataset, batch_sampler=sampler,
                                num_workers=opts.n_workers,
                                pin_memory=opts.pin_mem, collate_fn=collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=opts.n_workers, shuffle=False,
                                pin_memory=opts.pin_mem, collate_fn=collate_fn)
    dataloader = PrefetchLoader(dataloader)
    return dataloader


def build_aligned_dataloader(dataset_qa, dataset_qar, collate_fn, is_train, opts):
    batch_size = (opts.train_batch_size if is_train
                  else opts.val_batch_size)
    if is_train:
        sampler_qa = TokenBucketSampler([i + j for i, j in zip(dataset_qa.lens, dataset_qar.lens)],
                                        bucket_size=BUCKET_SIZE,
                                        batch_size=batch_size, droplast=is_train, seed=42)
        sampler_qar = TokenBucketSampler([i + j for i, j in zip(dataset_qa.lens, dataset_qar.lens)],
                                         bucket_size=BUCKET_SIZE,
                                         batch_size=batch_size, droplast=is_train, seed=42)
        dataloader_qa = DataLoader(dataset_qa, batch_sampler=sampler_qa,
                                   num_workers=opts.n_workers,
                                   pin_memory=opts.pin_mem, collate_fn=collate_fn)
        dataloader_qar = DataLoader(dataset_qar, batch_sampler=sampler_qar,
                                    num_workers=opts.n_workers,
                                    pin_memory=opts.pin_mem, collate_fn=collate_fn)

    else:
        dataloader_qa = DataLoader(dataset_qa, batch_size=batch_size,
                                   num_workers=opts.n_workers, shuffle=False,
                                   pin_memory=opts.pin_mem, collate_fn=collate_fn)
        dataloader_qar = DataLoader(dataset_qar, batch_size=batch_size,
                                    num_workers=opts.n_workers, shuffle=False,
                                    pin_memory=opts.pin_mem, collate_fn=collate_fn)
    dataloader_qa = PrefetchLoader(dataloader_qa)
    dataloader_qar = PrefetchLoader(dataloader_qar)
    return dataloader_qa, dataloader_qar


def build_optimizer(model, opts):
    """ vqa linear may get larger learning rate; bias, layer norm不加weight decay """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = [(n, p) for n, p in model.named_parameters()
                       if 'vcr_output' not in n]
    param_top = [(n, p) for n, p in model.named_parameters()
                 if 'vcr_output' in n]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_top
                    if not any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_top
                    if any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer


def load_img_feat(db_list, all_img_dbs, opts):
    db_ = db_list.split(";")
    assert len(db_) <= 2, "More than two img_dbs found"
    gt_db_path, db_path = "", ""
    for d in db_:
        if "gt" in d:
            gt_db_path = d
        else:
            db_path = d
    if gt_db_path != "":
        img_db_gt = DetectFeatLmdb(
            gt_db_path, -1, opts.max_bb, opts.min_bb, 100,
            opts.compressed_db)
        all_img_dbs.path2imgdb[gt_db_path] = img_db_gt
    else:
        img_db_gt = None
    img_db = all_img_dbs[db_path] if db_path != "" else None
    all_img_dbs.path2imgdb[db_path] = img_db
    return img_db, img_db_gt


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    # load DBs and image dirs
    all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
                                 opts.num_bb, opts.compressed_db)
    # train
    LOGGER.info(f"Loading Train Dataset "
                f"{opts.train_txt_dbs}, {opts.train_img_dbs}")
    train_datasets_qa = []
    train_datasets_qar = []
    for txt_path, img_path in zip(opts.train_txt_dbs, opts.train_img_dbs):
        img_db, img_db_gt = load_img_feat(img_path, all_img_dbs, opts)
        qa_txt_db = VcrTxtTokLmdb(txt_path, opts.max_txt_len, task="qa")
        qar_txt_db = VcrTxtTokLmdb(txt_path, opts.max_txt_len, task="qar")
        qa_txt_db.id2len = dict((k, qa_txt_db.id2len[k]) for k in qar_txt_db.id2len.keys())  # 保持样本数一致
        train_datasets_qa.append(
            VcrDataset(qa_txt_db, img_db_gt=img_db_gt, img_db=img_db))
        train_datasets_qar.append(
            VcrDataset(qar_txt_db, img_db_gt=img_db_gt, img_db=img_db))
    train_dataset_qa = ConcatDatasetWithLens(train_datasets_qa)
    train_dataset_qar = ConcatDatasetWithLens(train_datasets_qar)
    # train_dataloader = build_dataloader(train_dataset, vcr_collate, True, opts)
    train_dataloader_qa, train_dataloader_qar = build_aligned_dataloader(
        train_dataset_qa, train_dataset_qar, vcr_collate, True, opts)
    # val
    LOGGER.info(f"Loading Val Dataset {opts.val_txt_db}, {opts.val_img_db}")
    val_img_db, val_img_db_gt = load_img_feat(
        opts.val_img_db, all_img_dbs, opts)
    val_txt_db = VcrTxtTokLmdb(opts.val_txt_db, -1)
    val_dataset = VcrEvalDataset(
        "val", val_txt_db, img_db=val_img_db, img_db_gt=val_img_db_gt)
    val_final_dataset = VcrEvalDataset(
        "test", val_txt_db, img_db=val_img_db, img_db_gt=val_img_db_gt)
    val_dataloader = build_dataloader(val_dataset, vcr_eval_collate,
                                      False, opts)
    val_final_dataloader = build_dataloader(
        val_final_dataset, vcr_eval_collate,
        False, opts)

    all_dbs = opts.train_txt_dbs + [opts.val_txt_db]
    toker = json.load(open(f'{all_dbs[0]}/meta.json'))['bert']
    assert all(toker == json.load(open(f'{db}/meta.json'))['bert']
               for db in all_dbs)
    model = UniterForVisualCommonsenseReasoning.from_pretrained(
        opts.model_config, {}, img_dim=IMG_DIM)
    model.init_type_embedding()
    model.init_word_embedding(NUM_SPECIAL_TOKENS)
    # if opts.checkpoint_from == "vcr_pretrain":
    if opts.checkpoint is not None:
        opts.output_dir = os.path.dirname(opts.checkpoint)
        checkpoint = torch.load(opts.checkpoint)
        state_dict = checkpoint.get('model_state', checkpoint)
        matched_state_dict = {}
        unexpected_keys = set()
        missing_keys = set()
        for name, param in model.named_parameters():
            missing_keys.add(name)
        for key, data in state_dict.items():
            if key in missing_keys:
                matched_state_dict[key] = data
                missing_keys.remove(key)
            else:
                unexpected_keys.add(key)
        print("Unexpected_keys:", list(unexpected_keys))
        print("Missing_keys:", list(missing_keys))
        model.load_state_dict(matched_state_dict, strict=False)
    del checkpoint
    model.to(device)
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    model, optimizer = amp.initialize(model, optimizer,
                                      enabled=opts.fp16, opt_level='O2')
    val_log, results = validate(model, val_dataloader)


def compute_accuracies(out_qa, labels_qa, out_qar, labels_qar):
    outputs_qa = out_qa.max(dim=-1)[1]
    outputs_qar = out_qar.max(dim=-1)[1]
    matched_qa = outputs_qa.squeeze() == labels_qa.squeeze()
    matched_qar = outputs_qar.squeeze() == labels_qar.squeeze()
    matched_joined = matched_qa & matched_qar
    n_correct_qa = matched_qa.sum().item()
    n_correct_qar = matched_qar.sum().item()
    n_correct_joined = matched_joined.sum().item()
    return n_correct_qa, n_correct_qar, n_correct_joined


@torch.no_grad()
def validate(model, val_loader, align_fn=None, visualize=False):
    if hvd.rank() == 0:
        val_pbar = tqdm(total=len(val_loader))
    else:
        val_pbar = NoOp()
    LOGGER.info("start running validation...")
    model.eval()
    val_qa_loss, val_qar_loss = 0, 0
    tot_qa_score, tot_qar_score, tot_score = 0, 0, 0
    loss_align = 0
    n_ex = 0
    st = time()
    results = {}
    with h5py.File(os.path.join(args.output_dir, f'attention_{logitorprob}.h5'), 'w') as h5fp:
        for i, batch in enumerate(val_loader):
            # 分开qa和qar的输入部分
            batch_qa, batch_qar = {}, {}
            qids = batch['qids']
            for key, value in batch.items():
                if len(value) == len(qids)*8:
                    qa_idxs = list(cytoolz.concat([(8 * i, 8 * i + 1, 8 * i + 2, 8 * i + 3) for i in range(len(qids))]))
                    qar_idxs = list(cytoolz.concat([(8 * i + 4, 8 * i + 5, 8 * i + 6, 8 * i + 7) for i in range(len(qids))]))
                    batch_qa[key] = value[qa_idxs]
                    batch_qar[key] = value[qar_idxs]
                else:
                    batch_qa[key] = value
                    batch_qar[key] = value

            # scores = model(batch, compute_loss=False)
            drawer_qa = None
            drawer_qar = None
            # if visualize and n_ex % 1000 == 0:
            # from utils.visualize import get_attention_drawer
            #     dirname = os.path.join(LOGGER.handlers[0].baseFilename[:-12], 'figures')
            #     drawer_qa = get_attention_drawer(os.path.join(dirname, f'attention_map_qa_{n_ex}.png'))
            #     drawer_qar = get_attention_drawer(os.path.join(dirname, f'attention_map_qar_{n_ex}.png'))
            scores1, att_qa, att_qa_mask = model(batch_qa, compute_loss=False, output_attention=True, draw_attention=drawer_qa, logitorprob=logitorprob)
            scores2, att_qar, att_qar_mask = model(batch_qar, compute_loss=False, output_attention=True, draw_attention=drawer_qar, logitorprob=logitorprob)

            qa_targets = batch['qa_targets']
            qar_targets = batch['qar_targets']
            for _idx, qid in enumerate(qids):
                ex_group = h5fp.create_group(qid)
                q2a_group = ex_group.create_group('q2a')
                qa2r_group = ex_group.create_group('qa2r')
                # import pdb; pdb.set_trace()
                q2a_group.create_dataset('target', data=qa_targets[_idx, 0].item())
                qa2r_group.create_dataset('target', data=qar_targets[_idx, 0].item())
                for choice in range(4):
                    att_qa_ = torch.stack([att_qa[layer][_idx][choice] for layer in range(12)], dim=0) # layer x head x nobj
                    q2a_group.create_dataset(f'attention{choice}', data=att_qa_.data.cpu().numpy())
                    q2a_group.create_dataset(f'mask{choice}', data=att_qa_mask[_idx][choice][0].data.cpu().numpy())  # nobj

                    att_qar_ = torch.stack([att_qar[layer][_idx][choice] for layer in range(12)], dim=0) # layer x head x nobj
                    qa2r_group.create_dataset(f'attention{choice}', data=att_qar_.data.cpu().numpy())
                    qa2r_group.create_dataset(f'mask{choice}', data=att_qar_mask[_idx][choice][0].data.cpu().numpy())  # nobj

            # scores = scores.view(len(qids), -1)
            scores1 = scores1.view(len(qids), 4)
            scores2 = scores2.view(len(qids), 4)
            if align_fn is not None:
                loss_align_ = align_fn(
                    att_qa, att_qa_mask, qa_targets.squeeze(-1),
                    att_qar, att_qar_mask, qar_targets.squeeze(-1),
                )
                lw = F.sigmoid(model.layer_weights.float())
                hw = F.sigmoid(model.head_weights.float())
                loss_align_ = loss_align_ * lw.view(1, -1, 1) * hw.view(1, 1, -1)
                loss_align_ = loss_align_.mean()
                loss_align += loss_align_.item() * len(qids)
            vcr_qa_loss = F.cross_entropy(
                scores1, qa_targets.squeeze(-1), reduction="sum")
            # vcr_qa_loss = F.cross_entropy(
                    # scores[:, :4], qa_targets.squeeze(-1), reduction="sum")
            # if scores.shape[1] > 8:
            #     qar_scores = []
            #     for batch_id in range(scores.shape[0]):
            #         answer_ind = qa_targets[batch_id].item()
            #         qar_index = [4+answer_ind*4+i
            #                      for i in range(4)]
            #         qar_scores.append(scores[batch_id, qar_index])
            #     qar_scores = torch.stack(qar_scores, dim=0)
            # else:
            #     qar_scores = scores[:, 4:]
            # vcr_qar_loss = F.cross_entropy(
            #     qar_scores, qar_targets.squeeze(-1), reduction="sum")
            vcr_qar_loss = F.cross_entropy(
                scores2, qar_targets.squeeze(-1), reduction="sum")
            val_qa_loss += vcr_qa_loss.item()
            val_qar_loss += vcr_qar_loss.item()
            curr_qa_score, curr_qar_score, curr_score = compute_accuracies(
                scores1, qa_targets, scores2, qar_targets)
            # curr_qa_score, curr_qar_score, curr_score = compute_accuracies(
            #     scores[:, :4], qa_targets, qar_scores, qar_targets)
            tot_qar_score += curr_qar_score
            tot_qa_score += curr_qa_score
            tot_score += curr_score
            # for qid, score in zip(qids, scores):
            #     results[qid] = score.cpu().tolist()
            for qid, score1, score2 in zip(qids, scores1, scores2):
                results[qid] = score1.cpu().tolist() + score2.cpu().tolist()
            n_ex += len(qids)
            val_pbar.update(1)
        val_qa_loss = sum(all_gather_list(val_qa_loss))
        val_qar_loss = sum(all_gather_list(val_qar_loss))
        tot_qa_score = sum(all_gather_list(tot_qa_score))
        tot_qar_score = sum(all_gather_list(tot_qar_score))
        tot_score = sum(all_gather_list(tot_score))
        n_ex = sum(all_gather_list(n_ex))
        tot_time = time()-st
        val_qa_loss /= n_ex
        val_qar_loss /= n_ex
        val_qa_acc = tot_qa_score / n_ex
        val_qar_acc = tot_qar_score / n_ex
        val_acc = tot_score / n_ex
        if align_fn is not None:
            loss_align = sum(all_gather_list(loss_align))
            loss_align /= n_ex
        val_log = {'valid/vcr_qa_loss': val_qa_loss,
                   'valid/vcr_qar_loss': val_qar_loss,
                   'valid/acc_qa': val_qa_acc,
                   'valid/acc_qar': val_qar_acc,
                   'valid/acc': val_acc,
                   'valid/align_loss': loss_align,
                   'valid/ex_per_s': n_ex/tot_time}
        model.train()
        align_str = ''

        # if align_fn is not None:
        #     for key, value in align_fn.get_metrics(True).items():
        #         if 'acc' in key:
        #             align_str += f'{key}: {value*100:.2f};'
        #         else:
        #             align_str += f'{key}: {value:.2f};'
        LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                    f"score_qa: {val_qa_acc*100:.2f} "
                    f"score_qar: {val_qar_acc*100:.2f} "
                    f"score: {val_acc*100:.2f} "
                    f"loss_qa: {val_qa_loss:.2f} "
                    f"loss_qar: {val_qar_loss:.2f} "
                    f"loss_align: {loss_align:.2f}"
                    f"{align_str}")
    return val_log, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--model_config",
                        default=None, type=str,
                        help="json file for model architecture")
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size", default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size", default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--lr_mul", default=10.0, type=float,
                        help="multiplier for top layer lr")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=2.0, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=4000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for. (invsqrt decay)")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true', help="pin memory")

    # can use config files
    parser.add_argument('--alpha', type=float, default=1.0, help='weight of align loss')
    parser.add_argument('--gamma', type=float, default=1.0, help='weight of align coefficient')
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)

    if exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
