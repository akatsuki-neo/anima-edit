# 编辑数据集规则：
# - 训练图片存放在参数--train_data_dir目录下的各个子文件夹中
# - 子文件夹命名格式可设为「<重复次数>_<兜底标注文本>」，示例：`10_character_a`
#   名称前缀数字控制该目录数据重复训练次数，无前缀数字时默认重复次数为1
# - 单图配套txt标注优先级高于文件夹兜底标注。
#   例：存在`foo.png`和同名`foo.txt`时，优先使用foo.txt内容作为图片提示词。
# - 多参考图依靠文件主名匹配，匹配到的全部参考图会一并生效：
#   foo.png + foo_ref.png + foo_ref1.jpg + foo_ref2.webp
#     → 以foo.png作为训练目标，[foo_ref.png, foo_ref1.jpg, foo_ref2.webp]全部作为参考图
#   参考图的文件后缀可以和目标图不一致
# - 文件名以`_ref/_ref1/_ref2/……`结尾的文件仅作参考图，不会被当作训练目标参与训练
# - 没有匹配任何`*_ref*`参考文件的图片，默认按普通样本进行训练
# - 只有需要让无参考图的样本把自身当作参考图时，才添加启动参数--anima_self_reference_test
# - 元数据格式的数据集，也可通过 reference_image_paths / reference_images / refs 字段手动显式指定参考图片

accelerate launch anima_train_network.py \
  --pretrained_model_name_or_path /root/autodl-tmp/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors \
  --vae /root/autodl-tmp/ComfyUI/models/vae/qwen_image_vae.safetensors \
  --qwen3 /root/autodl-tmp/ckn/qwen3 \
  --t5_tokenizer_path configs/t5_old \
  --train_data_dir /root/autodl-tmp/ckn/downloaded_images \
  --output_dir out \
  --output_name anima_self_ref_overfit \
  --network_module networks.lora_anima \
  --network_dim 128 \
  --network_alpha 128 \
  --network_train_unet_only \
  --learning_rate 1e-4 \
  --optimizer_type AdamW \
  --max_train_steps 5000000 \
  --train_batch_size 2 \
  --mixed_precision bf16 \
  --save_precision bf16 \
  --resolution 1024,1024 \
  --enable_bucket \
  --caption_extension .txt \
  --anima_multi_image_edit \
  --anima_self_reference_test \
  --gradient_checkpointing \
  --log_with wandb \
  --logging_dir logs \
  --log_tracker_name anima-edit \
  --wandb_run_name anima_self_ref_overfit \
  --sample_every_n_steps 100 \
  --anima_sample_reference_dir /root/autodl-tmp/ckn/test \
