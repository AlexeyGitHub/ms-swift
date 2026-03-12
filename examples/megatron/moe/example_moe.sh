CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift export \
    --model /mnt/filesystem-n3/khoroshilov/models/qwen3-30b-a3b-base \
    --use_hf 1 \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir /mnt/filesystem-n3/khoroshilov/train/other_train/ms-swift/models_v2/Qwen3-30B-A3B-Base-mcore \