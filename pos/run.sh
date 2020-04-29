#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python -u main_pos_ud.py --exp_name bibert-124k-en-ar --bert_model bibert-124k --gpuid 0 --batchsize 8 --warmup_proportion 0.20000 --learning_rate 0.000020 --max_epoch 7 --target_language ar --source_language en --seed 0 --num_duplicate 20
