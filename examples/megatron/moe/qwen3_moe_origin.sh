# ZeRO3: 91.2s/it; 16 * 80GiB
# Megatron-LM: 9.6s/it; 16 * 60GiB
# Launch using Alibaba Cloud DLC
# https://help.aliyun.com/zh/pai/user-guide/general-environment-variables
# ref: https://github.com/modelscope/ms-swift/blob/main/examples/train/multi-node/dlc/train.sh
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --load /mnt/filesystem-n3/khoroshilov/train/other_train/ms-swift/models_v2/qwen3-coder-30b-a3b-instruct-mcore \
    --dataset 'liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT' \
    --tensor_model_parallel_size 4 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters 100 \
    --eval_iters 50 \
    --min_lr 1e-6 \
    --save /mnt/filesystem-n3/khoroshilov/train/other_train/ms-swift/megatron_output/qwen3-coder-30b-a3b-instruct-v6 \
    --eval_interval 100 \
    --save_interval 100 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --use_flash_attn true \
    --wandb_project sft-v5.3 \
    --wandb_exp_name qwen3-coder-30b-a3b-instruct-SFT-v6 \
    --sequence_parallel true
