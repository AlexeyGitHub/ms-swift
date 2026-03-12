NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model /mnt/filesystem-n3/khoroshilov/models/qwen3-30b-a3b-base \
    --train_type lora \
    --dataset 'swift/self-cognition#1000' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --router_aux_loss_coef 1e-3 \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 2048 \
    --output_dir output/qwen3-30b-a3b-base-v1 \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --deepspeed zero3 \
    --model_name qwen3-30b-a3b-base-v0