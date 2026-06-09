# anima-edit

大概流程和kohya ss的sd-scripts训练lora的流程是一样的

要用cui跑训练出来的东西的话用这个节点https://github.com/spawner1145/comfyui-adapters.git

# 首先, 数据集

传一个文件夹`--train_data_dir 你的文件夹路径`, 里面有多个子文件夹, 子文件夹前面是`2_`这种前缀代表这个子文件夹里面内容重复几次

子文件夹里面放图片和txt对

## 如何加参考图？
如果你要练`ccip_adapter`, 例如`train_ccip_adapter.sh`, 建议直接自指训练, 直接加`--anima_self_reference_test`这个命令行会自指训练

如果是latent参考训练如`train_latent.sh`, 不建议自指, 这里假设你的一个目标图和txt对是`114514.jpg`和`114514.txt`, 要加参考图就在那个子文件夹里面放个`114514_ref.jpg`这种, 在末尾加上`_ref`后缀

多图的话就按这样的顺序`_ref`、`_ref1`、`_ref2`......这种后缀依次作为第一个参考图第二个参考图和第三个参考图, 这些参考图和目标图以及txt文件为一组数据

# 接下来是正式训练

懒得看我说的就直接看三个train的sh脚本`train_full.sh`, `train_latent.sh`, `train_ccip_adapter.sh`, 功能分别是联合训练(所有都开), 只训练latent参考的lora, 以及训练`ccip adapter`(同时训练lora)

值得一提的是训练`ccip adapter`的时候可以选择不训练lora/lycoris, 在命令行中加入`--anima_disable_network_training`, 不过一般效果很烂不建议开

```bash
--anima_multi_image_edit   #是否开启latent参考
--anima_ip_adapter   #是否开启adapter
--anima_ip_adapter_weights out/name_ip_adapter.safetensors   #加载预训练的ccip adapter权重
--network_weights 你的lora或者lokr地址   #加载预训练的lora/lokr权重
```

## 注意事项
当你训练ccip adapter的时候, 这几个参数必开
```bash
--anima_ip_adapter \
--anima_train_ip_adapter \
--anima_sample_reference_dir /root/autodl-tmp/ckn/test \
--anima_ip_adapter_feature_backend ccip_tokens \
--anima_ip_adapter_feature_model /root/autodl-tmp/ckn/ccip_model/ccip-caformer_b36-24.ckpt \
--anima_ip_adapter_lr 5e-5 \
```

这里的ccip文件可以在这里拿到:https://huggingface.co/deepghs/ccip/blob/main/ccip-caformer_b36-24.ckpt

训练latent参考时, 开lora训练以及加`--anima_multi_image_edit`这个参数

关于lora, 感觉没什么好说的

标准lora：
```bash
--network_module networks.lora_anima \
--network_dim 128 \
--network_alpha 128 \
```

lokr:
```bash
--network_module networks.lokr \
--network_dim 114514 \
--network_alpha 114514 \
--network_args "factor=4" \
```

就看你喜欢挂什么, 我个人感觉没啥区别（

# 还有一点采样
众所周知我们训练会开wandb
```bash
--log_with wandb \
--logging_dir logs \
--log_tracker_name anima-edit \
--wandb_run_name anima_ipadapter_self_ref_overfit \
```

然后有时候想看效果, 首先先把必要参数写了
```bash
--sample_every_n_steps 200 \
--anima_sample_reference_dir /root/autodl-tmp/ckn/test \
```
这里这个test文件夹就可以放我们的参考图和txt文件对

关于命名和训练数据集是一样的，几个_ref后缀图片加一个txt文件

txt文件内部支持写参数，会自动解析
```bash
--w width
--h height
--s sample steps
--d seed
--l scale / CFG
--n negative prompt
--fs flow_shift
```

我有点懒得写文档了, 具体的看几个sh脚本吧, 都放在那里了（