#!/usr/bin/env bash

## run the training
python src/train.py \
--batch_size 32 \
--buffer_size 1000000 \
--min_replay_size 50000 \
--epsilon_decay 1000000 \
--target_update_frequency 10000 \
--save_interval 10000 \
--log_interval 1000 \
--gamma 0.99 \
--epsilon_start 1.0 \
--epsilons_end 0.1 \
--learning_rate 2.5e-4 \
--save_path './src/model/model.pack' \
--log_dir './src/logs' \