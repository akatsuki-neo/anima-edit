# anima-edit

See the upstream sd-scripts documentation for the common training options.

Anima edit options:

```bash
# Flux2 Klein-style multi-image reference token concat
--anima_multi_image_edit

# Enable the Anima IP-Adapter path. This does not enable token concat by itself.
--anima_ip_adapter

# Train and save IP-Adapter weights separately from LoRA/LyCORIS.
--anima_train_ip_adapter

# Load existing IP-Adapter weights before training/resume/fine-tune.
--anima_ip_adapter_weights out/name_ip_adapter.safetensors

# Optional dedicated IP-Adapter learning rate. Defaults to unet_lr or learning_rate.
--anima_ip_adapter_lr 1e-4

# Train IP-Adapter only, without LoRA/LyCORIS optimizer params.
--anima_train_ip_adapter \
--anima_disable_network_training
```

Training IP-Adapter feature backends:

```bash
# choices: vae, ccip, lsnet
--anima_ip_adapter_feature_backend vae

# required for ccip/lsnet
--anima_ip_adapter_feature_model /path/to/model_dir

# optional if the feature dim cannot be inferred from the backend model
--anima_ip_adapter_feature_dim 768

# projected tokens per CCIP/LSNet reference feature
--anima_ip_adapter_num_tokens 4
```

Backend model directories:

```text
ccip:  a directory containing metrics.json, model_feat.onnx, model_metrics.onnx
lsnet: a directory containing config.json and one .pth/.pt checkpoint
       config.json example: {"model": "lsnet_xl_artist_448"}
```

Saved weights:

```text
LoRA/LyCORIS: out/name.safetensors
IP-Adapter:  out/name_ip_adapter.safetensors
```

Inference:

```bash
python anima_minimal_inference.py \
  ... \
  --ip_adapter \
  --ip_adapter_weight out/name_ip_adapter.safetensors \
  --ip_adapter_feature_backend vae \
  --reference_image /path/ref1.png /path/ref2.png
```

Inference uses non-training option names:

```bash
--ip_adapter_feature_backend ccip \
--ip_adapter_feature_model /path/to/ccip_dir \
--ip_adapter_num_tokens 4
```

Use `--no_reference_sequence` at inference for IP-Adapter-only conditioning.
