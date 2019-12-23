#!/bin/bash

MODEL_RECOVER_PATH=/mnt/nlpdemo/docker_data/distill_bert
CONFIG_PATH=/mnt/nlpdemo/docker_data/distill_bert/bert_config.json
POS_DATA_DIR=/mnt/nlpdemo/docker_data/POS
NER_DATA_DIR=/mnt/nlpdemo/docker_data/NER/CoNLL-2003
CHUNKING_DATA_DIR=/mnt/nlpdemo/docker_data/chunking/conll2000
SRL_DATA_DIR=/mnt/nlpdemo/docker_data/SRL
OUTPUT_DIR=/mnt/nlpdemo/unilm-small-out

python examples/run_mtdnn_v3.py --model_type bert  --cache_dir bert_cache --model_name_or_path $MODEL_RECOVER_PATH  --do_lower_case --output_dir $OUTPUT_DIR --max_seq_length 128 --do_train --do_eval --per_gpu_train_batch_size 128 --per_gpu_eval_batch_size 128 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 4.0 --overwrite_cache --pos_data_dir $POS_DATA_DIR --ner_data_dir $NER_DATA_DIR --chunking_data_dir $CHUNKING_DATA_DIR --srl_data_dir $SRL_DATA_DIR --ft_before_eval --labels_srl $SRL_DATA_DIR/labels.txt --save_steps 1000 --overwrite_output_dir 

