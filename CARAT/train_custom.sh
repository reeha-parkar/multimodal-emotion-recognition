#!/bin/bash

# Training script for my custom unaligned MOSEI dataset retaining the original temporal intervals
# Uses the CustomNonAlignedMOSEI dataloader

torchrun --nproc_per_node=1 main.py \
    --do_train \
    --epochs=20 \
    --lr 5e-5 \
    --seed 1 \
    --visual_num_hidden_layers 4 \
    --text_num_hidden_layers 6 \
    --audio_num_hidden_layers 4 \
    --binary_threshold 0.25 \
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
    --unaligned_mask_same_length
