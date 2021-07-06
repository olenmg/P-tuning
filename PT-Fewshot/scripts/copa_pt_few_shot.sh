#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python3 cli.py \
--data_dir ../FewGLUE_32dev/COPA \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--task_name copa \
--output_dir ../output/copa \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 1 \
--pet_per_gpu_train_batch_size 8 \
--pet_gradient_accumulation_steps 2 \
--pet_max_seq_length 96 \
--pet_max_steps 3500 \
--pattern_ids 1 \
