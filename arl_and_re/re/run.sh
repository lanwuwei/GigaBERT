#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python main.py --exp_name bibert-124k-en-ar --bert_model bibert-124k  --gpuid 0 --batchsize 4 --warmup_proportion 0.20000 --learning_rate 0.00002 --max_epoch 7 --target_language ar --source_language en --seed 1234 --num_duplicate 20 --max_seq_length 512
