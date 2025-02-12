export WANDB_DISABLED=true

# 层是从 0 开始的
# 命名有点小问题，input_layers指的是input_layers结束的层号
# ctx_layers指的是ctx_layers结束的层号
# input_layers 到 ctx_layers 之间是预测 ctx token 的层
# ctx_layers 到最后是预测 word token 的层


# 参数说明：
# 总bz数 = per_device_train_batch_size * gradient_accumulation_steps * 实际运行卡数
# 每个epoch会存一次ckpt，包括 gpt2 和 codebook
# vae_pretrained_model_path 是传入已经训练好的 codebook 的参数，输入路径即可，codebook类型需要与 vae_config_path参数一致
# 这里的 chunk_size 会覆盖codebook config yaml中的 chunk_size，而且不能为空，需要一致的话改成一致就可以
# training_type 的类型
#   1. full：全部参数都会训练
#   2. codebook：只训练codebook，目前由于框架比较复杂不能完全对齐先前代码的训练结果，建议训练codebook暂时先用原先的，训练好后传进来
#   3. after_input_layer_include_cb：同时训练ctx predict层和word predict层，包括codebook
#   4. after_input_layer_exclude_cb：同时训练ctx predict层和word predict层，不包括codebook
#   5. except_codebook：训练除了codebook的所有层
#   6. only_output_layer：只训练word predict层
#   7. only_ctx_layer_include_cb：只训练ctx predict层，包括codebook
#   8. only_ctx_layer_exclude_cb：只训练ctx predict层，不包括codebook
#   9. ours：暴露出来DIY的接口，随便改的


python run_train_hlm.py \
    --dataset_name wiki_103_path \
    --dataset_config_name wikitext-103-raw-v1 \
    --model_name_or_path gpt2_model_path \
    --model_type gpt2 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --vae_config_path ../conf/models/residualvq_24_1024_512_kmeans.yaml \
    --vae_pretrained_model_path ../runs/residualvq_24_1024_512_kmeans_layer6_mlp_re/latest_checkpoint.pt \
    --input_layers 6 \
    --ctx_layers 7 \
    --do_train \
    --weight_decay=0.1 \
    --save_strategy "epoch" \
    --warmup_steps=900 \
    --lr_scheduler_type="cosine" \
    --learning_rate 1e-3 \
    --logging_steps 10 \
    --fp16 \
    --output_dir ./trained_models/ail_bz256_epo10_lr1e3_input6_output7_cs4_residualvq_24_1024_512_ckpt446_re \
    --overwrite_output_dir \
    --chunk_size 4 \
    --training_type full

