#!/bin/bash
export CUDA_VISIBLE_DEVICES=5

python3 cli.py \
--data_dir ../FewGLUE_32dev/WSC \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--task_name wsc \
--output_dir ../output/wsc \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 1 \
--pet_per_gpu_train_batch_size 4 \
--pet_gradient_accumulation_steps 4 \
--pet_max_seq_length 128 \
--pet_max_steps 3500 \
--pattern_ids 2 \
--learning_rate 1e-4
