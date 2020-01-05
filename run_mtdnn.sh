#!/bin/bash
MODEL_RECOVER_PATH=/data/hb_base_model/unilm_v2_model_2.bin
CONFIG_NAME=/data/hb_base_model/bert_config.json
POS_DATA_DIR=/data/POS/UD_English-EWT
NER_DATA_DIR=/data/NER/CoNLL-2003
TRAIN_FILE=/data/POS/UD_English-EWT/en_ewt-ud-train.conllu
PREDICT_FILE=/data/POS/UD_English-EWT/en_ewt-ud-test.conllu
OUTPUT_DIR=/data/model_pos_unilm
export CUDA_VISIBLE_DEVICES=3
python examples/run_mtdnn.py --pos_data_dir $POS_DATA_DIR --ner_data_dir $NER_DATA_DIR --gradient_accumulation_steps 4 --output_dir $OUTPUT_DIR --model_type bert --model_name_or_path bert-base-multilingual-cased  --max_seq_length 128 --do_train  --do_eval --per_gpu_train_batch_size 32 --num_train_epochs 3.0 --save_steps 1000 --overwrite_cache --overwrite_output_dir --weight_decay 0.01 --warmup_steps 80  --ft_before_eval
