# torchrun --nproc_per_node=1 main.py \
#     --do_train \
#     --epochs=20 \
#     --lr 5e-5 \
#     --seed 1 \
#     --visual_num_hidden_layers 4 \
#     --text_num_hidden_layers 6 \
#     --audio_num_hidden_layers 4 \
#     --binary_threshold 0.25 \
#     --recon_mse_weight 1.0 \
#     --aug_mse_weight 1.0 \
#     --beta_mse_weight 0.0 \
#     --lsr_clf_weight 0.01 \
#     --recon_clf_weight 0.0 \
#     --aug_clf_weight 0.1 \
#     --shuffle_aug_clf_weight 0.1 \
#     --total_aug_clf_weight 1.0 \
#     --cl_weight 1.0 \
#     --aligned \
#     --data_path './data/train_valid_test.pt'


torchrun --nproc_per_node=1 main.py \
    --do_train \
    --epochs=20 \
    --lr 5e-5 \
    --seed 1 \
    --aligned \
    --data_path './data/train_valid_test.pt'


#python -m main.py --do_train --epochs=20  --lr 5e-5 --gpu_id '1' --seed 1 --visual_num_hidden_layers 4 --text_num_hidden_layers 6 --audio_num_hidden_layers 4 --binary_threshold 0.25 --recon_mse_weight 1.0 --aug_mse_weight 1.0 --beta_mse_weight 0.0 --lsr_clf_weight 0.01 --recon_clf_weight 0.0 --aug_clf_weight 0.1 --shuffle_aug_clf_weight 0.1 --total_aug_clf_weight 1.0 --cl_weight 1.0 --aligned --data_path './data/train_valid_test.pt' --local_rank 0

#python -m torch.distributed.launch --nproc_per_node=1 main.py --do_train --epochs=20  --lr 5e-5 --gpu_id '1' --seed 1 --visual_num_hidden_layers 4 --text_num_hidden_layers 6 --audio_num_hidden_layers 4 --binary_threshold 0.25 --recon_mse_weight 1.0 --aug_mse_weight 1.0 --beta_mse_weight 0.0 --lsr_clf_weight 0.01 --recon_clf_weight 0.0 --aug_clf_weight 0.1 --shuffle_aug_clf_weight 0.1 --total_aug_clf_weight 1.0 --cl_weight 1.0 --aligned --data_path './data/train_valid_test.pt --local_rank 0'