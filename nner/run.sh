#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python -u main_ner.py --exp_name bibert-124k-en-ar --bert_model bibert-124k --gpuid 0 --batchsize 8 --warmup_proportion 0.40000 --learning_rate 0.000100 --max_epoch 7 --target_language ar --source_language en --seed 0 --num_duplicate 64
