export TORCH_NCCL_ENABLE_MONITORING=0
export GLOO_SOCKET_IFNAME= eth0
export TP_SOCKET_IFNAME=eth0
export NCLL_SOCKET_IFNAME=eth0

MODEL="/data/vpc-intern/linliqunbj/model/Qwen3-30B-A3B-mcore"
DATASET1="/data/vpc-intern/linliqunbj/data/swift/apolo_math_text_data_clean_v1_rm_repeat.json"
OUTPUT_DIR="/data/vpc-intern/linliqunbj/Qwen3-30B-A3B-mcore_output"

NNODES=$WORLD_SIZE
NODE_RANK=$RANK
NPROC_PER_NODE=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
megatron sft
--load $MODEL
--dataset $DATASET1
--tensor_model_parallel_size 2
--expert_model_parallel_size 8
--moe_grouped_gemm true
--moe_shared_expert_overlap true
--moe_aux_loss_coeff 0.01
--micro_batch_size 1
--global_batch_size 16
--packing true
--recompute_granularity full
--recompute_method uniform
--recompute_num_layers 1
--train_iters 2000
--eval_iters 50
--finetune true
--cross_entropy_loss_fusion true
--lr 1e-5
--lr_warmup_iters 100
--min_lr 1e-6
--save megatron_output/Qwen3-30B-A3B-Base
--eval_interval 200
--save_interval 200
--max_length 8192
--num_workers 8
--dataset_num_proc 8
--no_save_optim true
--no_save_rng true
--sequence_parallel true
--use_flash_attn true