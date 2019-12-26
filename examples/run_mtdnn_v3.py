from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from utils_ner import convert_examples_to_features as convert_examples_to_features_ner
from utils_ner import get_labels as get_labels_ner
from utils_ner import read_examples_from_file as read_examples_from_file_ner


from utils_pos import convert_examples_to_features as convert_examples_to_features_pos
from utils_pos import get_labels as get_labels_pos
from utils_pos import read_examples_from_file as read_examples_from_file_pos

from utils_chunking import convert_examples_to_features as convert_examples_to_features_chunking
from utils_chunking import get_labels as get_labels_chunking
from utils_chunking import read_examples_from_file as read_examples_from_file_chunking

from utils_srl import convert_examples_to_features as convert_examples_to_features_srl
from utils_srl import get_labels as get_labels_srl
from utils_srl import read_examples_from_file as read_examples_from_file_srl

import torch.nn as nn
from torch.optim import Adam
import copy
import requests

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer
from transformers import MTDNNModelv3 as MTDNNModel
from transformers import MTDNNModelTaskEmbeddingV2 as TaskEmbeddingModel
from transformers import AdapterMTDNNModel 
from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizer
from transformers import DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer



softmax = nn.Softmax(dim=0)
num_layers = 12

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, DistilBertConfig)),
    ())

MODEL_CLASSES = {
    "bert":(BertConfig, MTDNNModel, BertTokenizer),
    "task_embedding":(BertConfig, TaskEmbeddingModel, BertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def finetune(args, train_dataset, model, tokenizer, labels, pad_token_label_id, task="pos"):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    alpha_sets = ["alpha_pos", "alpha_ner", "alpha_chunking", "alpha_srl"]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in (no_decay + alpha_sets))],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in alpha_sets)], 'lr': args.alpha_learning_rate}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.ft_learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running task-specific finetune *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    
    do_alpha = args.do_alpha


    if task == "pos":
        task_id = 0
        layer_id = args.layer_id_pos
    elif task == "ner":
        task_id = 1
        layer_id = args.layer_id_ner
    elif task == "chunking":
        task_id = 2
        layer_id = args.layer_id_chunking
    elif task == "srl":
        task_id = 3
        layer_id = args.layer_id_srl

    if args.ft_with_last_layer:
        do_alpha=False
        layer_id = -1

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3], 
                      "task_id": task_id,
                      "layer_id":layer_id,
                      "do_alpha": do_alpha}
            if args.model_type != "distilbert":
                inputs["token_type_ids"]: batch[2] if args.model_type in ["bert", "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if do_alpha or args.model_type.lower() == "task_embedding":
                alpha = outputs[0]
                # alpha_loss = outputs[1]
                loss = outputs[1]

                # loss = loss + alpha_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                # if args.do_alpha:
                #     alpha_loss = alpha_loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                # if args.do_alpha:
                #     alpha_loss = alpha_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % 100 == 0 :
                
                print("loss", loss.item())
                if args.model_type.lower() == "task_embedding":
                    print("alpha", alpha[:num_layers])

                elif args.do_alpha:
                    alpha_pos = softmax(model.module.alpha_pos).detach().cpu().numpy()[:num_layers]
                    alpha_ner = softmax(model.module.alpha_ner).detach().cpu().numpy()[:num_layers]
                    alpha_chunking = softmax(model.module.alpha_chunking).detach().cpu().numpy()[:num_layers]
                    alpha_srl = softmax(model.module.alpha_srl).detach().cpu().numpy()[:num_layers]
                    print("alpha_pos", alpha_pos)
                    print("alpha_ner", alpha_ner)
                    print("alpha_chunking", alpha_chunking)
                    print("alpha_srl", alpha_srl)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
  
    return global_step, tr_loss / global_step, model


def load_and_cache_dev_examples(args, tokenizer, pos_labels, ner_labels, chunking_labels, srl_labels, pad_token_label_id, is_ft=False):

    # Load data features from cache or dataset file
    
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    if is_ft:
        prefix = "dev_ft"
    else:
        prefix = "dev"

    cached_features_file = os.path.join(args.pos_data_dir, "cached_{}_{}_{}".format(prefix,
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length)))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        cached_features = torch.load(cached_features_file)
        pos_features, ner_features, chunking_features, srl_features = cached_features

    else:       
    
        logger.info("Creating pos features from dataset file at %s", args.pos_data_dir)
        if is_ft:
            pos_examples = read_examples_from_file_pos(args.pos_data_dir, "train")
        else:
            pos_examples = read_examples_from_file_pos(args.pos_data_dir, "dev")
        pos_features = convert_examples_to_features_pos(pos_examples, pos_labels, args.max_seq_length, tokenizer,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ["roberta"]),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ["xlnet"]),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                pad_token_label_id=pad_token_label_id
                                                )
        logger.info("Creating ner features from dataset file at %s", args.ner_data_dir)
        if is_ft:
            ner_examples = read_examples_from_file_ner(args.ner_data_dir, "train")
        else:   
            ner_examples = read_examples_from_file_ner(args.ner_data_dir, "dev")
        ner_features = convert_examples_to_features_ner(ner_examples, ner_labels, args.max_seq_length, tokenizer,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ["roberta"]),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ["xlnet"]),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                pad_token_label_id=pad_token_label_id
                                                )
        
        logger.info("Creating chunking features from dataset file at %s", args.ner_data_dir)
        if is_ft:
            chunking_examples = read_examples_from_file_chunking(args.chunking_data_dir, "train")
        else:   
            chunking_examples = read_examples_from_file_chunking(args.chunking_data_dir, "dev")
        chunking_features = convert_examples_to_features_chunking(chunking_examples, chunking_labels, args.max_seq_length, tokenizer,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ["roberta"]),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ["xlnet"]),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                pad_token_label_id=pad_token_label_id
                                                )
        
        logger.info("Creating srl features from dataset file at %s", args.srl_data_dir)
        if is_ft:
            srl_examples = read_examples_from_file_srl(args.srl_data_dir, "train")
        else:
            srl_examples = read_examples_from_file_srl(args.srl_data_dir, "test")
        
        srl_features = convert_examples_to_features_srl(srl_examples, srl_labels, args.max_seq_length, tokenizer,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ["roberta"]),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ["xlnet"]),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                pad_token_label_id=pad_token_label_id)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            features = (pos_features, ner_features, chunking_features, srl_features)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in pos_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in pos_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in pos_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in pos_features], dtype=torch.long)

    pos_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    all_input_ids = torch.tensor([f.input_ids for f in ner_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in ner_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in ner_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in ner_features], dtype=torch.long)

    ner_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    all_input_ids = torch.tensor([f.input_ids for f in chunking_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in chunking_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in chunking_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in chunking_features], dtype=torch.long)

    chunking_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    all_input_ids = torch.tensor([f.input_ids for f in srl_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in srl_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in srl_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in srl_features], dtype=torch.long)

    srl_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return pos_dataset, ner_dataset, chunking_dataset, srl_dataset

def load_and_cache_train_examples(args, tokenizer, pos_labels, ner_labels, chunking_labels, srl_labels, pad_token_label_id):

    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    cached_features_file = os.path.join(args.pos_data_dir, "cached_{}_{}_{}".format("mtTrain",
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length)))
    
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        cached_features = torch.load(cached_features_file)
        pos_features, ner_features, chunking_features, srl_features = cached_features
    else:
        # load POS dataset
        logger.info("Creating POS features from dataset file at %s", args.pos_data_dir)
        pos_examples = read_examples_from_file_pos(args.pos_data_dir, "train")
        pos_features = convert_examples_to_features_pos(pos_examples, pos_labels, args.max_seq_length, tokenizer,
                                                    cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                    # xlnet has a cls token at the end
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=bool(args.model_type in ["roberta"]),
                                                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                    pad_on_left=bool(args.model_type in ["xlnet"]),
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                    pad_token_label_id=pad_token_label_id
                                                    )
        # load NER dataset
        logger.info("Creating NER features from dataset file at %s", args.ner_data_dir)
        ner_examples = read_examples_from_file_ner(args.ner_data_dir, "train")
        ner_features = convert_examples_to_features_ner(ner_examples, ner_labels, args.max_seq_length, tokenizer,
                                                    cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                    # xlnet has a cls token at the end
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=bool(args.model_type in ["roberta"]),
                                                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                    pad_on_left=bool(args.model_type in ["xlnet"]),
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                    pad_token_label_id=pad_token_label_id
                                                    )
        
        # load chunking dataset
        logger.info("Creating Chunking features from dataset file at %s", args.ner_data_dir)
        chunking_examples = read_examples_from_file_chunking(args.chunking_data_dir, "train")
        chunking_features = convert_examples_to_features_chunking(chunking_examples, chunking_labels, args.max_seq_length, tokenizer,
                                                    cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                    # xlnet has a cls token at the end
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=bool(args.model_type in ["roberta"]),
                                                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                    pad_on_left=bool(args.model_type in ["xlnet"]),
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                    pad_token_label_id=pad_token_label_id
                                                    )
        
        # load srl dataset
        logger.info("Creating srl features from dataset file at %s", args.srl_data_dir)
        srl_examples = read_examples_from_file_srl(args.srl_data_dir, "train")
        srl_features = convert_examples_to_features_srl(srl_examples, srl_labels, args.max_seq_length, tokenizer,
                                                    cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                    # xlnet has a cls token at the end
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=bool(args.model_type in ["roberta"]),
                                                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                    pad_on_left=bool(args.model_type in ["xlnet"]),
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                    pad_token_label_id=pad_token_label_id
                                                    )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            features = (pos_features, ner_features, chunking_features, srl_features)
            torch.save(features, cached_features_file)



    # pack the dataset t into mini-batch: Dt
    mini_batch_size = args.per_gpu_train_batch_size
    cnt = 0
    pos_features_batchs = []
    while cnt + mini_batch_size < len(pos_features):
        batch_t = []
        for i in range(cnt, cnt + mini_batch_size):
            batch_t.append(pos_features[i])
        pos_features_batchs.append(batch_t)
        cnt += mini_batch_size
    batch_t = []
    for i in range(cnt, len(pos_features)):
        batch_t.append(pos_features[i])
    pos_features_batchs.append(batch_t)

    ner_features_batchs = []
    cnt = 0
    while cnt + mini_batch_size < len(ner_features):
        batch_t = []
        for i in range(cnt, cnt + mini_batch_size):
            batch_t.append(ner_features[i])
        ner_features_batchs.append(batch_t)
        cnt += mini_batch_size
    batch_t = []
    for i in range(cnt, len(ner_features)):
        batch_t.append(ner_features[i])
    ner_features_batchs.append(batch_t)

    chunking_features_batches = []
    cnt = 0
    while cnt + mini_batch_size < len(chunking_features):
        batch_t = []
        for i in range(cnt, cnt + mini_batch_size):
            batch_t.append(chunking_features[i])
        chunking_features_batches.append(batch_t)
        cnt += mini_batch_size
    batch_t = []
    for i in range(cnt, len(chunking_features)):
        batch_t.append(chunking_features[i])
    chunking_features_batches.append(batch_t)

    srl_features_batches = []
    cnt = 0
    while cnt + mini_batch_size < len(srl_features):
        batch_t = []
        for i in range(cnt, cnt + mini_batch_size):
            batch_t.append(srl_features[i])
        srl_features_batches.append(batch_t)
        cnt += mini_batch_size
    batch_t = []
    for i in range(cnt, len(srl_features)):
        batch_t.append(srl_features[i])
    srl_features_batches.append(batch_t)

    data_list = [pos_features_batchs, ner_features_batchs, chunking_features_batches, srl_features_batches]

    return data_list


def train(args, train_data_list, model, tokenizer, labels_pos, labels_ner, labels_chunking, labels_srl, pad_token_label_id):
    """ Train the model """ 
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    t_total = sum(len(x) for x in train_data_list) // args.gradient_accumulation_steps * args.num_train_epochs

    # prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    alpha_sets = ['alpha_pos', 'alpha_ner', 'alpha_chunking', 'alpha_srl']
    # srl_sets = ['alpha_srl']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in (no_decay + alpha_sets))], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in alpha_sets)], 'lr':args.alpha_learning_rate}
        ]
    

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(" Num Epochs = %d", args.num_train_epochs)
    logger.info(" Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args)
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    alpha_log = "alpha.log.training"
    alpha_log_f = open(alpha_log, "w+", encoding="utf-8")

    # for each epoch, 1. merge all the datasets, 2. shuffle
    step = 0
    for _ in train_iterator:
        train_data_list = [sorted(t, key=lambda k:random.random()) for t in train_data_list]
        all_iters = [iter(item) for item in train_data_list]
        all_indices = []
        all_indices = [0]*len(train_data_list[0]) + [1]*len(train_data_list[1]) + [2]*len(train_data_list[2]) + [3]*len(train_data_list[3])
        random.shuffle(all_indices)
        model.train()
        epoch_iterator = tqdm(all_indices, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, task_id in enumerate(epoch_iterator):
            features = next(all_iters[task_id])
            input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(args.device)
            input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(args.device)
            segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(args.device)
            label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long).to(args.device)

            if task_id == 0:
                layer_id = args.layer_id_pos
            elif task_id == 1:
                layer_id = args.layer_id_ner
            elif task_id == 2:
                layer_id = args.layer_id_chunking
            elif task_id == 3:
                layer_id = args.layer_id_srl

            inputs = {"input_ids":input_ids, 
                      "attention_mask":input_mask, 
                      "token_type_ids":segment_ids,
                      "labels":label_ids, 
                      "task_id":task_id, 
                      "layer_id":layer_id,
                      "do_alpha":args.do_alpha}
            outputs = model(**inputs)
            loss = outputs[0]
            if args.model_type.lower() == "task_embedding":
                alpha = outputs[0]
                loss = outputs[1]
            
            elif args.do_alpha:
                loss = outputs[1]
                # alpha_loss = outputs[1]
                # alpha = outputs[0]     
                # loss = loss + alpha_loss
            # scale loss
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            tr_loss += loss.item()
            # print("loss", loss.item())

            if (step + 1 ) % 100 == 0:
                
                print("loss", loss.item())
                if args.model_type.lower() == "task_embedding":
                    print("alpha", alpha[:num_layers])

                elif args.do_alpha:
                    alpha_pos = softmax(model.module.alpha_pos).detach().cpu().numpy()[:num_layers]
                    alpha_ner = softmax(model.module.alpha_ner).detach().cpu().numpy()[:num_layers]
                    alpha_chunking = softmax(model.module.alpha_chunking).detach().cpu().numpy()[:num_layers]
                    alpha_srl = softmax(model.module.alpha_srl).detach().cpu().numpy()[:num_layers]
                    print("alpha_pos", alpha_pos)
                    print("alpha_ner", alpha_ner)
                    print("alpha_chunking", alpha_chunking)
                    print("alpha_srl", alpha_srl)

                    alpha_log_f.write(str(step+1))
                    alpha_log_f.write(" ".join([str(x) for x in alpha_pos.reshape(len(alpha_pos))]) + "\n")
                    alpha_log_f.write(" ".join([str(x) for x in alpha_ner.reshape(len(alpha_ner))]) + "\n")
                    alpha_log_f.write(" ".join([str(x) for x in alpha_chunking.reshape(len(alpha_chunking))]) + "\n")
                    alpha_log_f.write(" ".join([str(x) for x in alpha_srl.reshape(len(alpha_srl))]) + "\n")
                    alpha_log_f.write('\n')
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1 

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(args, model, tokenizer, labels_pos, labels_ner, pad_token_label_id, mode="dev")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    alpha_log_f.close()
    return global_step, tr_loss / global_step, model

def evaluate(args, model, tokenizer, eval_dataset, labels, pad_token_label_id, mode, prefix=" ", task="pos"):

    if torch.cuda.device_count() > 0:
        eval_batch_size = torch.cuda.device_count() * args.per_gpu_eval_batch_size
    else:
        eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    logger.info("***** Running  {} evaluation  *****".format(task))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    if task == "pos":
        task_id = 0
        layer_id = args.layer_id_pos
    elif task == "ner":
        task_id = 1
        layer_id = args.layer_id_ner
    elif task == "chunking":
        task_id = 2
        layer_id = args.layer_id_chunking
    elif task == "srl":
        task_id = 3
        layer_id = args.layer_id_srl

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids":batch[2],
                      "labels": batch[3], 
                      "task_id":task_id, 
                      "layer_id": layer_id,
                      "do_alpha": args.do_alpha}
            if args.model_type != "distilbert":
                inputs["token_type_ids"]: batch[2] if args.model_type in ["bert", "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            if args.do_alpha:
                alpha = outputs[0]
                outputs = outputs[1:]
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
    
    results = {}
    if task == "pos":
        results["pos_accuracy"] = accuracy_score(out_label_list, preds_list)
    elif task == "ner":
        results = {
        # "ner_loss": eval_loss,
        # "ner_precision": precision_score(out_label_list, preds_list),
        # "ner_recall": recall_score(out_label_list, preds_list),
        "ner_f1": f1_score(out_label_list, preds_list)
    }
    elif task == "chunking":
        results = {
            "chunking_f1": f1_score(out_label_list, preds_list)
        }
    elif task == "srl":
        results = {
            "srl_f1": f1_score(out_label_list, preds_list)
        }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    
    
    
    parser.add_argument("--pos_data_dir", type=str, default="")
    parser.add_argument("--ner_data_dir", type=str, default="")
    parser.add_argument("--chunking_data_dir", type=str, default="")
    parser.add_argument("--srl_data_dir", type=str, default="")

    parser.add_argument("--ft_before_eval", action="store_true")
    parser.add_argument("--layer_id_pos", type=int, default=-1)
    parser.add_argument("--layer_id_ner", type=int, default=-1)
    parser.add_argument("--layer_id_chunking", type=int, default=-1)
    parser.add_argument("--layer_id_srl", type=int, default=-1)
    parser.add_argument("--labels_srl", type=str)

    parser.add_argument("--alpha_learning_rate", type=float, default=1e-3)
    parser.add_argument("--init_last", action="store_true")
    parser.add_argument("--ft_learning_rate", type=float, default=5e-5)

    parser.add_argument("--do_alpha", action="store_true")
    parser.add_argument("--ft_with_last_layer", action="store_true")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--use_adapter", action="store_true")
    parser.add_argument("--send_msg", action="store_true")
    parser.add_argument("--task_description", type=str, help="for collecting results")
    args = parser.parse_args()


    
    layer_id_pos = args.layer_id_pos
    layer_id_ner = args.layer_id_ner
    layer_id_chunking = args.layer_id_chunking
    layer_id_srl = args.layer_id_srl


    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare CONLL-2003 task for NER, Universe Dependency for pos-tag, CONLL-2000 task for chunking
    labels_ner = get_labels_ner(args.labels)
    num_labels_ner = len(labels_ner)
    labels_pos = get_labels_pos(args.labels)
    num_labels_pos = len(labels_pos)
    labels_chunking = get_labels_chunking(args.labels)
    num_labels_chunking = len(labels_chunking)
    labels_srl = get_labels_srl(args.labels_srl)
    num_labels_srl = len(labels_srl)
    
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels_ner,
                                          cache_dir=args.cache_dir if args.cache_dir else None,
                                          output_hidden_states=True)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config, 
                                        num_labels_pos=num_labels_pos, 
                                        num_labels_ner=num_labels_ner,
                                        num_labels_chunking=num_labels_chunking,
                                        num_labels_srl=num_labels_srl,
                                        cache_dir=args.cache_dir if args.cache_dir else None,
                                        init_last=args.init_last)
    num_layers = config.num_hidden_layers
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_train_examples(args, tokenizer, labels_pos, labels_ner, labels_chunking, labels_srl, pad_token_label_id)
        # print("dataset lens", len(train_dataset))
        # logger.info("first dataset lens :{}".format(type(train_dataset[0])))
        global_step, tr_loss, _ = train(args, train_dataset, model, tokenizer, labels_pos, labels_ner, labels_chunking, labels_srl, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:

        # do 2 stage evaluation: finetune , then evaluate

        # step 1: finetune and on ner task


        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        msg_dict = {}
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, num_labels_pos=num_labels_pos, num_labels_ner=num_labels_ner, num_labels_chunking=num_labels_chunking, num_labels_srl=num_labels_srl)
            model.to(args.device)
        

            pos_dataset_ft, ner_dataset_ft, chunking_dataset_ft, srl_dataset_ft = load_and_cache_dev_examples(args, tokenizer, labels_pos, labels_ner, labels_chunking, labels_srl, pad_token_label_id, is_ft=True)
            pos_dataset, ner_dataset, chunking_dataset, srl_dataset = load_and_cache_dev_examples(args, tokenizer, labels_pos, labels_ner, labels_chunking, labels_srl, pad_token_label_id, is_ft=False)
            
            logger.info("Evaluate before finetune")

            result_pos_no_ft, _ = evaluate(args, model, tokenizer, pos_dataset, labels_pos, pad_token_label_id, mode="dev", prefix=global_step, task="pos")
            result_ner_no_ft, _ = evaluate(args, model, tokenizer, ner_dataset, labels_ner, pad_token_label_id, mode="dev", prefix=global_step, task="ner")
            result_chunking_no_ft, _ = evaluate(args, model, tokenizer, chunking_dataset, labels_chunking, pad_token_label_id, mode="dev", prefix=global_step, task="chunking")
            result_srl_no_ft, _ = evaluate(args, model, tokenizer, srl_dataset, labels_srl, pad_token_label_id, mode="dev", task="srl")

            msg_dict["pos_no_ft"] = result_pos_no_ft["pos_accuracy"]
            msg_dict["ner_no_ft"] = result_ner_no_ft["ner_f1"]
            msg_dict["chunking_no_ft"] = result_chunking_no_ft["chunking_f1"]
            msg_dict["srl_no_ft"] = result_srl_no_ft["srl_f1"]

            
            torch.save(model, "source_model.pl")    
            logger.info("Finetuning and Evaluate")

            # POS tag
            _, _, model = finetune(args, pos_dataset_ft, model, tokenizer, labels_pos, pad_token_label_id, task="pos")
            result, _ = evaluate(args, model, tokenizer, pos_dataset, labels_pos, pad_token_label_id, mode="dev", prefix=global_step, task="pos")

            msg_dict["pos_after_ft"] = result["pos_accuracy"] 
            # NER
            model = torch.load("source_model.pl")
            _, _, model = finetune(args, ner_dataset_ft, model, tokenizer,  labels_ner, pad_token_label_id, task="ner")
            result, _ = evaluate(args, model, tokenizer, ner_dataset, labels_ner, pad_token_label_id, mode="dev", prefix=global_step, task="ner")
           
            msg_dict["ner_after_ft"] = result["ner_f1"]

            # Chunking
            model = torch.load("source_model.pl")
            _, _, model = finetune(args, chunking_dataset_ft, model, tokenizer,  labels_chunking, pad_token_label_id, task="chunking")
            result, _ = evaluate(args, model, tokenizer, chunking_dataset, labels_chunking, pad_token_label_id, mode="dev", prefix=global_step, task="chunking")

            msg_dict["chunking_after_ft"] = result["chunking_f1"]

            # SRL
            model = torch.load("source_model.pl")
            _, _, model = finetune(args, srl_dataset_ft, model, tokenizer,  labels_srl, pad_token_label_id, task="srl")
            result, _ = evaluate(args, model, tokenizer, srl_dataset, labels_srl, pad_token_label_id, mode="dev", prefix=global_step, task="srl")

            msg_dict["srl_after_ft"] = result["srl_f1"]

        

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
        
        if args.send_msg:
            api = "https://sc.ftqq.com/SCU47715T1085ec82936ebfe2723aaa3095bb53505ca315d2865a0.send"
            
            text = "+ ".join([str(key) + " : " + str(msg_dict[key]) + "\n" for key in msg_dict])
            data = {
                "text": args.task_description,
                "desp": text

            }
            print(text)
            requests.post(api, data=data)



    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.output_dir)
        model.to(args.device)
        result, predictions = evaluate(args, model, tokenizer, labels_pos, labels_ner,pad_token_label_id, mode="test")
        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(args.data_dir, "test.txt"), "r") as f:
                example_id = 0
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        writer.write(line)
                        if not predictions[example_id]:
                            example_id += 1
                    elif predictions[example_id]:
                        output_line = line.split()[0] + " " + predictions[example_id].pop(0) + "\n"
                        writer.write(output_line)
                    else:
                        logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])

    return results


if __name__ == "__main__":
    main()






            
            






    






            
            

