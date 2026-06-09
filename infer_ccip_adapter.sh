# 注意没训练latent参考的时候这个必开--no_reference_sequence

python anima_minimal_inference.py \
  --dit /root/autodl-tmp/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors \
  --vae /root/autodl-tmp/ComfyUI/models/vae/qwen_image_vae.safetensors \
  --text_encoder /root/autodl-tmp/ckn/qwen3 \
  --prompt "1girl, blue background, commentary, facing to the side, finger to face, highres, looking at viewer, makoto mirai academy high school uniform, pointing, pointing up, translation request, upper body, yano akane" \
  --image_size 1024 1024 \
  --reference_image /root/autodl-tmp/ckn/test/0d4634a46bab35b9e310dacd350e15d8.jpg \
  --save_path out/infer.png \
  --lora_weight out/anima_ipadapter_self_ref_overfit-step00028500.safetensors \
  --lora_multiplier 1.0 \
  --ip_adapter \
  --omni-adapter \
  --ip_adapter_weight out/anima_ipadapter_self_ref_overfit-step00028500_ip_adapter.safetensors \
  --ip_adapter_feature_backend ccip_tokens \
  --ip_adapter_feature_model /root/autodl-tmp/ckn/anima_edit/ccip-caformer_b36-24.ckpt \
  --no_reference_sequence \
  --infer_steps 30 \
  --guidance_scale 3.5