#!/bin/bash

swift export \
    --model /mnt/filesystem-n3/khoroshilov/models/qwen3-coder-30b-a3b-instruct \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir /mnt/filesystem-n3/khoroshilov/train/other_train/ms-swift/models_v2/qwen3-coder-30b-a3b-instruct-mcore \
    --test_convert_precision true