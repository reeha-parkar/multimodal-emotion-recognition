#!/bin/bash

# Training script for my custom unaligned MOSEI dataset retaining the original temporal intervals

torchrun --nproc_per_node=1 main.py \
    --do_train \
    --epochs=15 \
    --lr 3e-5 \
    --warmup_proportion 0.2 \
    --lr_decay 0.98 \
    --seed 1 \
    --batch_size 64 \
    --proto_m 0.99 \
    --moco_queue 8192 \
    --gradient_accumulation_steps 1 \
    --visual_num_hidden_layers 3 \
    --text_num_hidden_layers 4 \
    --audio_num_hidden_layers 3 \
    --binary_threshold 0.20 \
    --recon_mse_weight 1.0 \
    --aug_mse_weight 1.0 \
    --beta_mse_weight 0.0 \
    --lsr_clf_weight 0.01 \
    --recon_clf_weight 0.0 \
    --aug_clf_weight 0.1 \
    --shuffle_aug_clf_weight 0.1 \
    --total_aug_clf_weight 1.0 \
    --cl_weight 1.0 \
    --use_custom_dataset \
    --custom_data_path './data/cmu_mosei_unaligned_ree.pt' \
    --ctc_target_length 400 \
    --text_max_position_embeddings 600 \
    --visual_max_position_embeddings 1200 \
    --audio_max_position_embeddings 1200 \
    --hidden_size 512 \
    --num_thread_reader 4 \
    --n_display 50 \
    --local-rank 0