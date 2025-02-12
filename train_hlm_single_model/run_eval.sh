export WANDB_DISABLED=true

#!/bin/bash

# 定义模型列表（可以替换为你的模型路径）
MODEL_LIST=(
  "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output9_cs4_residualvq_24_1024_512_ckpt446_re_t/checkpoint-446"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output9_cs4_residualvq_24_1024_512_ckpt446_re_t/checkpoint-893"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output9_cs4_residualvq_24_1024_512_ckpt446_re_t/checkpoint-1339"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output9_cs4_residualvq_24_1024_512_ckpt446_re_t/checkpoint-1786"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output9_cs4_residualvq_24_1024_512_ckpt446_re_t/checkpoint-2232"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output9_cs4_residualvq_24_1024_512_ckpt446_re_t/checkpoint-2679"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output9_cs4_residualvq_24_1024_512_ckpt446_re_t/checkpoint-3125"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output9_cs4_residualvq_24_1024_512_ckpt446_re_t/checkpoint-3572"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output9_cs4_residualvq_24_1024_512_ckpt446_re_t/checkpoint-4018"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/ail_bz256_epo10_lr1e3_input6_output9_cs4_residualvq_24_1024_512_ckpt446_re_t/checkpoint-4460"

 )


# 数据集路径
DATASET_NAME="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/datasets/wikitext103"
DATASET_CONFIG_NAME="wikitext-103-raw-v1"
OUTPUT_DIR_BASE="checkpoints"

# 循环遍历模型列表
for MODEL in "${MODEL_LIST[@]}"; do
  # 输出当前模型路径
  echo "Evaluating model: ${MODEL}"

  # 运行评估脚本
  CUDA_VISIBLE_DEVICES=0 python -u run_clm_eval.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --output_dir "${OUTPUT_DIR_BASE}/${MODEL//\//_}" \
    --do_eval \
    --eval_subset test \
    --input_layers 6 \
    --ctx_layers 9 \
    --vae_config_path ../conf/models/residualvq_24_1024_512_kmeans.yaml \
    --stride 1024 \
    --chunk_size 4

  echo "Evaluation for ${MODEL} completed."
done
