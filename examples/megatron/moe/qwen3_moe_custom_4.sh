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
    --split_dataset_ratio 0.01 \
    --sequence_parallel true \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 1 \
    --expert_model_parallel_size 2 \
    --context_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --transformer_impl transformer_engine \
    --fp8_format hybrid \
    --fp8_recipe delayed \
    --fp8_amax_history_len 1024 \
    --fp8_amax_compute_algo max \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 4 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save /mnt/filesystem-n3/khoroshilov/train/other_train/ms-swift/megatron_output/qwen3-coder-30b-a3b-instruct-v5 \
    --eval_interval 100 \
    --save_interval 100 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --wandb_project sft-v5.3 \
    --wandb_exp_name qwen3-coder-30b-a3b-instruct-SFT-v5 \
    --log_throughput true \
    --log_params_norm true | tee /mnt/filesystem-n3/khoroshilov/train/other_train/ms-swift/logs/training_output.log
