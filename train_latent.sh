# Dataset rules:
# - Put images under subfolders of --train_data_dir.
# - A subfolder may be named <repeat>_<class tokens>, for example 10_character_a.
#   The numeric prefix controls repeats; without the prefix, repeat defaults to 1.
# - Per-image captions win over folder class tokens: foo.png uses foo.txt when it exists.
# - Multiple references are matched by filename stem and are all used together:
#   foo.png + foo_ref.png + foo_ref1.jpg + foo_ref2.webp
#   trains foo.png as the target with [foo_ref.png, foo_ref1.jpg, foo_ref2.webp] as references.
#   Reference extensions may differ from the target extension.
# - Files whose stem ends with _ref, _ref1, _ref2, ... are reference-only and not targets.
# - Images without matched references train as normal target-only samples by default.
# - Add --anima_self_reference_test only when target-only samples should use themselves as references.
# - Metadata datasets may also specify references via reference_image_paths / reference_images / refs.

accelerate launch anima_train_network.py \
  --pretrained_model_name_or_path /root/autodl-tmp/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors \
  --vae /root/autodl-tmp/ComfyUI/models/vae/qwen_image_vae.safetensors \
  --qwen3 /root/autodl-tmp/ckn/qwen3 \
  --t5_tokenizer_path configs/t5_old \
  --train_data_dir /root/autodl-tmp/ckn/anima-edit-action-transfer-12-akatsuki-vlm \
  --output_dir out_edit \
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
  --min_bucket_reso 256 \
  --max_bucket_reso 2048 \
  --bucket_reso_steps 16 \
  --caption_extension .txt \
  --anima_multi_image_edit \
  --gradient_checkpointing \
  --log_with wandb \
  --logging_dir logs \
  --log_tracker_name anima-edit \
  --wandb_run_name anima_self_ref_overfit \
  --sample_every_n_steps 50 \
  --anima_sample_reference_dir /root/autodl-tmp/ckn/etest \
  --save_every_n_steps 200 \
