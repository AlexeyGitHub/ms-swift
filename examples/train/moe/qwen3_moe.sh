# If you don't want to train the router, set:
# `--target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /mnt/filesystem-n3/khoroshilov/models/qwen3-coder-30b-a3b-instruct \
    --train_type full \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT#2000' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --router_aux_loss_coef 1e-3 \
    --gradient_accumulation_steps 8 \
    --packing true \
    --rope_scaling yarn \
    --max_length 32000 \
    --max_model_len 32000 \
    --eval_steps 100 \
    --save_steps 200 \
    --save_total_limit 30 \
    --logging_steps 1 \
    --max_length 2048 \
    --save_only_model true \
    --output_dir /mnt/filesystem-n3/khoroshilov/train/other_train/ms-swift/output/qwen3-coder-30b-a3b-instruct-v1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --use_liger_kernel true \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --report_to wandb \
    --sequence_parallel_size 4 \
    --model_name qwen3-coder-30b-a3b-instruct-SFT-v1
