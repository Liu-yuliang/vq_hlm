#!/bin/bash

#SBATCH --account=yfliu3
#SBATCH --job-name=residualvq
#SBATCH --partition=RTX3090,RTX4090,A100 # 用sinfo命令可以看到所有队列
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡或多进程，请调整此参数
#SBATCH --cpus-per-task=16  # 每个进程的CPU数量
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2
#SBATCH --output=./runs/residualvq/%j.out
#SBATCH --error=./runs/residualvq/%j.err

# CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python train_vq.py \
#  --data_config ./conf/data/layer6_mlp.yaml \
#  --ckpt_dir ./runs/residualvq_nonorm_ckpt5e3_layer10_epo10 \
#  --model_config conf/models/residualvq.yaml \
#  --lr 5e-3 \
#  --train_epochs 1 \
# #  --data_config \
# #  --model_config \

#   "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-446"
#   "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-892"
#   "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-1339"
#   "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-1785"
#   "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-2231"
#   "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-2678"
#   "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-3124"
#   "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-3571"
#   "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-4017"
#   "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-4460"
# CUDA_VISIBLE_DEVICES=1 
# train_vq_meanpooling

# train_vq_last
CUDA_VISIBLE_DEVICES=1 python train_vq_meanpooling_use_train_hlm_code.py \
 --ckpt_dir /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_cb_bz256_epo10_lr1e3_input6_output7_cs4_residualvq_24_1024_512_kmeans_pre_vq/checkpoint-4460 \
 --model_config conf/models/residualvq_24_1024_512_kmeans.yaml \
 --data_config ./conf/data/layer6_mlp.yaml \
 --chunk_size 4 \
 --test

# CUDA_VISIBLE_DEVICES=1 python train_vq_meanpooling_use_train_hlm_code.py \
#  --ckpt_dir /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/cb_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-892 \
#  --model_config conf/models/residualvq_512.yaml \
#  --data_config ./conf/data/layer6_mlp.yaml \
#  --chunk_size 4 \
#  --test


# CUDA_VISIBLE_DEVICES=1 python train_vq_meanpooling_use_train_hlm_code.py \
#  --ckpt_dir /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/cb_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-1339 \
#  --model_config conf/models/residualvq_512.yaml \
#  --data_config ./conf/data/layer6_mlp.yaml \
#  --chunk_size 4 \
#  --test


# CUDA_VISIBLE_DEVICES=1 python train_vq_meanpooling_use_train_hlm_code.py \
#  --ckpt_dir /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/cb_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-1785 \
#  --model_config conf/models/residualvq_512.yaml \
#  --data_config ./conf/data/layer6_mlp.yaml \
#  --chunk_size 4 \
#  --test

# CUDA_VISIBLE_DEVICES=1 python train_vq_meanpooling_use_train_hlm_code.py \
#  --ckpt_dir /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/cb_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-2231 \
#  --model_config conf/models/residualvq_512.yaml \
#  --data_config ./conf/data/layer6_mlp.yaml \
#  --chunk_size 4 \
#  --test

# CUDA_VISIBLE_DEVICES=1 python train_vq_meanpooling_use_train_hlm_code.py \
#  --ckpt_dir /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/cb_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-2678 \
#  --model_config conf/models/residualvq_512.yaml \
#  --data_config ./conf/data/layer6_mlp.yaml \
#  --chunk_size 4 \
#  --test

# CUDA_VISIBLE_DEVICES=1 python train_vq_meanpooling_use_train_hlm_code.py \
#  --ckpt_dir /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/cb_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-3124 \
#  --model_config conf/models/residualvq_512.yaml \
#  --data_config ./conf/data/layer6_mlp.yaml \
#  --chunk_size 4 \
#  --test

# CUDA_VISIBLE_DEVICES=1 python train_vq_meanpooling_use_train_hlm_code.py \
#  --ckpt_dir /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/cb_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-3571 \
#  --model_config conf/models/residualvq_512.yaml \
#  --data_config ./conf/data/layer6_mlp.yaml \
#  --chunk_size 4 \
#  --test

# CUDA_VISIBLE_DEVICES=1 python train_vq_meanpooling_use_train_hlm_code.py \
#  --ckpt_dir /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/cb_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-4017 \
#  --model_config conf/models/residualvq_512.yaml \
#  --data_config ./conf/data/layer6_mlp.yaml \
#  --chunk_size 4 \
#  --test

# CUDA_VISIBLE_DEVICES=1 python train_vq_meanpooling_use_train_hlm_code.py \
#  --ckpt_dir /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/cb_bz256_epo10_lr1e3_input6_output7_chunksize4_residualvq_512/checkpoint-4460 \
#  --model_config conf/models/residualvq_512.yaml \
#  --data_config ./conf/data/layer6_mlp.yaml \
#  --chunk_size 4 \
#  --test

