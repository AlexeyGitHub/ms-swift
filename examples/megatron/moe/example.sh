CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model /mnt/filesystem-n3/khoroshilov/models/qwen2.5-7b-instruct \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir /mnt/filesystem-n3/khoroshilov/train/other_train/ms-swift/models_v2/Qwen2.5-7B-Instruct-mcore \
    --test_convert_precision true