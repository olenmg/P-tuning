#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

python3 cli.py \
--data_dir ../FewGLUE_32dev/RTE \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--task_name rte \
--output_dir ../output_aft/rte \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 16 \
--pet_per_gpu_train_batch_size 1 \
--pet_gradient_accumulation_steps 16 \
--pet_max_seq_length 256 \
--pet_max_steps 3500 \
--warmup_steps 150 \
--pattern_ids 1 \
--learning_rate 1e-4
