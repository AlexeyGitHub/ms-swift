PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_LAUNCH_BLOCKING=1 \
NCCL_DEBUG=INFO \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
NCCL_SHM_DISABLE=0 \
NCCL_SOCKET_IFNAME=eth0 \
NCCL_DEBUG_SUBSYS=INIT,COMM,COLL \
NCCL_ABORT_ON_ERROR=0 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --load /mnt/filesystem-n3/khoroshilov/train/other_train/ms-swift/models_v2/Qwen3-30B-A3B-Base-mcore \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-en#500' \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --train_iters 2000 \
    --eval_iters 50 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-30B-A3B-Base-tp-v5 \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash