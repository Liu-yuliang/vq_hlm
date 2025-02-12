配置环境后，使用该目录下的transformers替换安装的即可

目前只测试了RVQ的训练，加其他模型应该很容易。没有跑过全套，可能需要有时间的同学全流程测试一下修改一些小bug

[the pile](../../vq_hlm/train_hlm_single_model/run_train_thepile.py)的训练也支持，但目前只能支持python ..py启动，还没有写对应脚本，原框架支持streaming有问题，自己写的显存分配有问题，gpu mem拉不满，建议暂时不要使用（多卡跑倒是可以跑）



对transformers主要改动的文件为：

train_hlm_single_model/transformers/models/gpt2/modeling_gpt2.py

train_hlm_single_model/transformers/modeling_utils.py

train_hlm_single_model/transformers/trainer.py

train_hlm_single_model/transformers/training_args.py
