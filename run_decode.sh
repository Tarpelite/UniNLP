#!/bin/bash
labels_pos=/mnt/nlpdemo/UniNLP/labels/pos.txt
labels_ner=/mnt/nlpdemo/UniNLP/labels/ner.txt
labels_chunking=/mnt/nlpdemo/UniNLP/labels/chunking.txt
labels_srl=/mnt/nlpdemo/UniNLP/labels/srl.txt

model_path=/mnt/nlpdemo/unilm-small-out/pytorch_model.bin
config_name=/mnt/nlpdemo/unilm-small-out/config.json
tokenizer_name=/mnt/nlpdemo/unilm-small-out/vocab.txt

python examples/run_mtdnn_decoder.py --model_type bert --model_name_or_path $model_path --labels_pos $labels_pos --labels_ner $labels_ner --labels_chunking $labels_chunking --labels_srl $labels_srl --config_name $config_name --tokenizer_name $tokenizer_name --do_lower_case 
