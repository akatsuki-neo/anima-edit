# 开--no_reference_sequence是因为训练没带latent特征

python anima_minimal_inference.py \
  --dit /root/autodl-tmp/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors \
  --vae /root/autodl-tmp/ComfyUI/models/vae/qwen_image_vae.safetensors \
  --text_encoder /root/autodl-tmp/ckn/qwen3 \
  --t5_tokenizer_path configs/t5_old \
  --prompt "your prompt here" \
  --reference_image /root/autodl-tmp/ckn/test/foo.png \
  --image_size 1024 1024 \
  --infer_steps 30 \
  --guidance_scale 3.5 \
  --ip_adapter \
  --ip_adapter_weight out/anima_ipadapter_self_ref_overfit-000000500_ip_adapter.safetensors \
  --ip_adapter_feature_backend ccip \
  --ip_adapter_feature_model /root/autodl-tmp/ckn/ccip_model \
  --ip_adapter_num_tokens 8 \
  --save_path outputs \
  --no_reference_sequence \