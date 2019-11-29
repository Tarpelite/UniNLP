#!/bin/bash
DATA_DIR=/data/NER/CoNLL-2003
OUTPUT_DIR=/data/model_unilm_ner_base
MODEL_RECOVER_PATH=/data/hb_base_model/unilm_v2_model_2.bin
export CUDA_VISIBLE_DEVICES=3
python biunilm/run_ner.py --do_train --do_eval --fp16 --amp --num_workers 24 \
  --bert_model bert-large-cased --new_segment_ids  \
  --data_dir ${DATA_DIR} --src_file eng.train.openNLP --tgt_file train.tgt.10k --eval_file eng.testa\
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 128 --max_position_embeddings 128 \
  --trunc_seg a --always_truncate_tail --max_len_b 16 \
  --mask_prob 0.7 --max_pred 16 \
  --train_batch_size 128 --gradient_accumulation_steps 1 \
  --learning_rate 0.00001 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 3
