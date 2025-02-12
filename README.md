

使用runner_rvq_test_use_train_hlm_code.sh可以从带gpt的ckpt中抽出来codebook部分单独eval其rec loss

使用runner_rvq_dlc_6_scheduler_1_2.sh可以在训练时进行lr schedule，但目前只支持在对应py中修改scheduer


train_hlm_single_model里还有一层readme，这里是本次运行添加的feature

1. example.sh里添加了训练参数说明

2. 暂时支持了pile，但还不完善
  
3. 支持了在GPT-VQVAE的整体ckpt中抽出来VQVAE单独eval
  
4. 支持了VQVAE加scheduler的训练

