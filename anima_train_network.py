# Anima LoRA training script

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from accelerate import Accelerator
from safetensors.torch import load_file, save_file
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import (
    anima_models,
    anima_train_utils,
    anima_utils,
    flux_train_utils,
    qwen_image_autoencoder_kl,
    sd3_train_utils,
    strategy_anima,
    strategy_base,
    train_util,
)
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class AnimaIPAdapterFeatureExtractor:
    def __init__(self, backend: str, model_dir: Optional[str], device: torch.device, dtype: torch.dtype):
        self.backend = backend
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.model = None
        self.transform = None
        self.feature_dim = None
        if backend == "ccip":
            if model_dir is None:
                raise ValueError("--anima_ip_adapter_feature_model is required for CCIP feature extraction.")
            from ccip_lib.ccip import ccip_batch_extract_features

            self.extract_fn = ccip_batch_extract_features
            self.feature_dim = self._infer_ccip_dim(model_dir)
        elif backend == "lsnet":
            if model_dir is None:
                raise ValueError("--anima_ip_adapter_feature_model is required for LSNet feature extraction.")
            self.model, self.transform, self.feature_dim = self._load_lsnet(model_dir, device)

    @staticmethod
    def _infer_ccip_dim(model_dir: str) -> Optional[int]:
        try:
            from ccip_lib.ccip import _open_feat_model

            session = _open_feat_model(model_dir)
            shape = session.get_outputs()[0].shape
            if len(shape) >= 2 and isinstance(shape[-1], int):
                return shape[-1]
        except Exception as e:
            logger.warning(f"Could not infer CCIP feature dim from {model_dir}: {e}")
        return None

    @staticmethod
    def _find_lsnet_checkpoint(model_dir: str) -> str:
        candidates = sorted(Path(model_dir).glob("*.pth")) + sorted(Path(model_dir).glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No .pth/.pt checkpoint found in LSNet model dir: {model_dir}")
        return str(candidates[0])

    @staticmethod
    def _load_lsnet(model_dir: str, device: torch.device):
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from lsnet_lib.inference_artist import (
            load_checkpoint_state,
            load_model,
            normalize_state_dict_keys,
            resolve_feature_dim,
            resolve_num_classes,
        )
        from lsnet_lib.lsnet_model.lsnet_artist import default_cfgs_artist  # noqa: F401

        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        model_name = config["model"]
        checkpoint = AnimaIPAdapterFeatureExtractor._find_lsnet_checkpoint(model_dir)
        state_dict = normalize_state_dict_keys(load_checkpoint_state(checkpoint))
        feature_dim = resolve_feature_dim(None, state_dict)
        args = SimpleNamespace(
            model=model_name,
            checkpoint=checkpoint,
            num_classes=resolve_num_classes(None, None, state_dict),
            feature_dim=feature_dim,
            mode="cluster",
            allow_head_reinit=True,
            device=str(device),
        )
        model = load_model(args, state_dict)
        model.to(device)
        model.eval()
        input_size = 448 if model_name.endswith("_448") else 224
        try:
            from lsnet_lib.lsnet_model.lsnet_artist import default_cfgs_artist

            if model_name in default_cfgs_artist:
                input_size = default_cfgs_artist[model_name].get("input_size", (3, input_size, input_size))[1]
        except Exception:
            pass
        data_config = resolve_data_config({"input_size": (3, input_size, input_size)}, model=model)
        transform = create_transform(**data_config)
        return model, transform, feature_dim

    def extract(self, image_paths: list[str]) -> torch.Tensor:
        if not image_paths:
            return torch.zeros((0, self.feature_dim or 0), dtype=self.dtype, device=self.device)
        if self.backend == "ccip":
            features = self.extract_fn(image_paths, model=self.model_dir)
            features = torch.from_numpy(np.asarray(features, dtype=np.float32))
            return features.to(device=self.device, dtype=self.dtype)
        if self.backend == "lsnet":
            tensors = [self.transform(Image.open(path).convert("RGB")) for path in image_paths]
            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                features = self.model(batch, return_features=True)
            return features.to(dtype=self.dtype)
        raise ValueError(f"Unsupported IP-Adapter feature backend: {self.backend}")


class AnimaNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        if args.fp8_base or args.fp8_base_unet:
            logger.warning("fp8_base and fp8_base_unet are not supported. / fp8_baseとfp8_base_unetはサポートされていません。")
            args.fp8_base = False
            args.fp8_base_unet = False
        args.fp8_scaled = False  # Anima DiT does not support fp8_scaled

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning("cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled")
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert train_dataset_group.is_text_encoder_output_cacheable(
                cache_supports_dropout=True
            ), "when caching Text Encoder output, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"

        args.anima_use_reference_sequence = getattr(args, "anima_multi_image_edit", False)

        if getattr(args, "anima_train_ip_adapter", False):
            args.anima_ip_adapter = True

        if getattr(args, "anima_multi_image_edit", False):
            assert not args.cache_latents and not args.cache_latents_to_disk, (
                "--anima_multi_image_edit currently encodes reference images during training, so it cannot be used with "
                "--cache_latents or --cache_latents_to_disk yet."
            )

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing"

        if args.unsloth_offload_checkpointing:
            if not args.gradient_checkpointing:
                logger.warning("unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled")
                args.gradient_checkpointing = True
            assert (
                not args.cpu_offload_checkpointing
            ), "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing"
            assert (
                args.blocks_to_swap is None or args.blocks_to_swap == 0
            ), "blocks_to_swap is not supported with unsloth_offload_checkpointing"

        train_dataset_group.verify_bucket_reso_steps(16)  # WanVAE spatial downscale = 8 and patch size = 2
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(16)

    def load_target_model(self, args, weight_dtype, accelerator):
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0

        # Load Qwen3 text encoder (tokenizers already loaded in get_tokenize_strategy)
        logger.info("Loading Qwen3 text encoder...")
        qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
        qwen3_text_encoder.eval()

        # Load VAE
        logger.info("Loading Anima VAE...")
        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae, device="cpu", disable_mmap=True, spatial_chunk_size=args.vae_chunk_size, disable_cache=args.vae_disable_cache
        )
        vae.to(weight_dtype)
        vae.eval()

        # Return format: (model_type, text_encoders, vae, unet)
        return "anima", [qwen3_text_encoder], vae, None  # unet loaded lazily

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, list[nn.Module]]:
        loading_dtype = None if args.fp8_scaled else weight_dtype
        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device
        ip_feature_dim = args.anima_ip_adapter_feature_dim
        if getattr(args, "anima_ip_adapter", False) and args.anima_ip_adapter_feature_backend != "vae":
            extractor = AnimaIPAdapterFeatureExtractor(
                args.anima_ip_adapter_feature_backend,
                args.anima_ip_adapter_feature_model,
                accelerator.device,
                weight_dtype,
            )
            self.ip_adapter_feature_extractor = extractor
            ip_feature_dim = ip_feature_dim or extractor.feature_dim
            if ip_feature_dim is None:
                raise ValueError(
                    "Could not infer IP-Adapter feature dim. Please pass --anima_ip_adapter_feature_dim."
                )
        else:
            self.ip_adapter_feature_extractor = None

        attn_mode = "torch"
        if args.xformers:
            attn_mode = "xformers"
        if args.attn_mode is not None:
            attn_mode = args.attn_mode

        # Load DiT
        logger.info(f"Loading Anima DiT model with attn_mode={attn_mode}, split_attn: {args.split_attn}...")
        model = anima_utils.load_anima_model(
            accelerator.device,
            args.pretrained_model_name_or_path,
            attn_mode,
            args.split_attn,
            loading_device,
            loading_dtype,
            args.fp8_scaled,
            enable_ip_adapter=getattr(args, "anima_ip_adapter", False),
            ip_adapter_scale=getattr(args, "anima_ip_adapter_scale", 1.0),
            ip_adapter_feature_dim=ip_feature_dim,
            ip_adapter_num_tokens=getattr(args, "anima_ip_adapter_num_tokens", 4),
        )

        # Store unsloth preference so that when the base NetworkTrainer calls
        # dit.enable_gradient_checkpointing(cpu_offload=...), we can override to use unsloth.
        # The base trainer only passes cpu_offload, so we store the flag on the model.
        self._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing

        # Block swap
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        return model, text_encoders

    def get_tokenize_strategy(self, args):
        # Load tokenizers from paths (called before load_target_model, so self.qwen3_tokenizer isn't set yet)
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.qwen3,
            t5_tokenizer_path=args.t5_tokenizer_path,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
        )
        return tokenize_strategy

    def get_tokenizers(self, tokenize_strategy: strategy_anima.AnimaTokenizeStrategy):
        return [tokenize_strategy.qwen3_tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_anima.AnimaLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check)

    def get_text_encoding_strategy(self, args):
        return strategy_anima.AnimaTextEncodingStrategy()

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        ip_params = list(self.get_ip_adapter_parameters(unet))
        if getattr(args, "anima_ip_adapter_weights", None):
            if len(ip_params) == 0:
                raise ValueError("--anima_ip_adapter_weights requires --anima_ip_adapter so IP-Adapter layers exist.")
            self.load_ip_adapter_weights(unet, args.anima_ip_adapter_weights)

        if not getattr(args, "anima_train_ip_adapter", False):
            for param in ip_params:
                param.requires_grad_(False)
            return

        if len(ip_params) == 0:
            raise ValueError("--anima_train_ip_adapter requires --anima_ip_adapter so IP-Adapter layers exist.")

        for param in ip_params:
            param.requires_grad_(True)

        original_prepare = network.prepare_optimizer_params_with_multiple_te_lrs
        original_save = network.save_weights
        original_load = network.load_weights

        def prepare_optimizer_params_with_ip(text_encoder_lr, unet_lr, default_lr):
            results = original_prepare(text_encoder_lr, unet_lr, default_lr)
            if isinstance(results, tuple):
                trainable_params, lr_descriptions = results
            else:
                trainable_params, lr_descriptions = results, None
            if getattr(args, "anima_disable_network_training", False):
                trainable_params = []
                lr_descriptions = [] if lr_descriptions is not None else None
                network.requires_grad_(False)
            lr = args.anima_ip_adapter_lr if args.anima_ip_adapter_lr is not None else (unet_lr or default_lr)
            trainable_params.append({"params": ip_params, "lr": lr})
            if lr_descriptions is not None:
                lr_descriptions.append(f"anima_ip_adapter:{lr}")
            return (trainable_params, lr_descriptions) if lr_descriptions is not None else trainable_params

        def save_weights_with_ip(file, dtype, metadata):
            if not getattr(args, "anima_disable_network_training", False):
                original_save(file, dtype, metadata)
            if getattr(args, "anima_train_ip_adapter", False):
                ip_file = self.get_ip_adapter_output_path(file)
                self.save_ip_adapter_weights(unet, ip_file, dtype, metadata)

        def remove_old_ip_checkpoint(file):
            if not getattr(args, "anima_train_ip_adapter", False):
                return
            ip_file = self.get_ip_adapter_output_path(file)
            if os.path.exists(ip_file):
                logger.info(f"removing old Anima IP-Adapter checkpoint: {ip_file}")
                os.remove(ip_file)

        def load_weights_with_ip(file):
            info = original_load(file)
            ip_file = self.get_ip_adapter_output_path(file)
            if os.path.exists(ip_file):
                self.load_ip_adapter_weights(unet, ip_file)
            return info

        network.prepare_optimizer_params_with_multiple_te_lrs = prepare_optimizer_params_with_ip
        network.save_weights = save_weights_with_ip
        network.load_weights = load_weights_with_ip
        network.remove_old_ip_checkpoint = remove_old_ip_checkpoint
        if getattr(args, "anima_train_ip_adapter", False):
            network.save_ip_adapter_state = lambda output_dir: self.save_ip_adapter_weights(
                unet,
                self.get_ip_adapter_state_path(output_dir),
                None,
                {"ss_anima_ip_adapter_state": "true"},
            )
            network.load_ip_adapter_state = lambda input_dir: (
                self.load_ip_adapter_weights(unet, self.get_ip_adapter_state_path(input_dir))
                if os.path.exists(self.get_ip_adapter_state_path(input_dir))
                else None
            )

    @staticmethod
    def get_ip_adapter_parameters(unet):
        for name, param in unet.named_parameters():
            if ".ip_adapter." in name:
                yield param

    def set_ip_adapter_trainable(self, unet, trainable: bool):
        for param in self.get_ip_adapter_parameters(unet):
            param.requires_grad_(trainable)

    @staticmethod
    def get_ip_adapter_state_dict(unet):
        return {
            name: tensor.detach().cpu()
            for name, tensor in unet.state_dict().items()
            if ".ip_adapter." in name
        }

    @staticmethod
    def get_ip_adapter_output_path(file):
        root, ext = os.path.splitext(file)
        return f"{root}_ip_adapter{ext}"

    @staticmethod
    def get_ip_adapter_state_path(output_dir):
        return os.path.join(output_dir, "anima_ip_adapter.safetensors")

    def save_ip_adapter_weights(self, unet, file, dtype, metadata):
        state_dict = self.get_ip_adapter_state_dict(unet)
        if dtype is not None:
            state_dict = {key: value.to(dtype=dtype) for key, value in state_dict.items()}
        metadata = dict(metadata or {})
        metadata["ss_anima_ip_adapter"] = "true"
        save_file(state_dict, file, metadata)
        logger.info(f"saved Anima IP-Adapter weights: {file}")

    def load_ip_adapter_weights(self, unet, file):
        logger.info(f"loading Anima IP-Adapter weights: {file}")
        state_dict = load_file(file, device="cpu")
        info = unet.load_state_dict(state_dict, strict=False)
        unexpected = [key for key in info.unexpected_keys if ".ip_adapter." not in key]
        if unexpected:
            raise ValueError(f"Unexpected keys while loading IP-Adapter: {unexpected}")
        return info

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            return None  # no text encoders needed for encoding
        return text_encoders

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, False
            )
        return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # We cannot move DiT to CPU because of block swap, so only move VAE
                logger.info("move vae to cpu to save memory")
                org_vae_device = vae.device
                vae.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info("move text encoder to gpu")
            text_encoders[0].to(accelerator.device)

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # cache sample prompts
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}")

                tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

                prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"  cache TE outputs for: {p}")
                                tokens_and_masks = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy, text_encoders, tokens_and_masks
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            # move text encoder back to cpu
            logger.info("move text encoder back to cpu")
            text_encoders[0].to("cpu")

            if not args.lowram:
                logger.info("move vae back to original device")
                vae.to(org_vae_device)

            clean_memory_on_device(accelerator.device)
        else:
            # move text encoder to device for encoding during training/validation
            text_encoders[0].to(accelerator.device)

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]  # compatibility
        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        qwen3_te = te[0] if te is not None else None

        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        anima_train_utils.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            unet,
            vae,
            qwen3_te,
            tokenize_strategy,
            text_encoding_strategy,
            self.sample_prompts_te_outputs,
        )

    @staticmethod
    def get_ip_adapter_norms(unet):
        param_sq = 0.0
        grad_sq = 0.0
        has_grad = False
        with torch.no_grad():
            for name, param in unet.named_parameters():
                if ".ip_adapter." not in name:
                    continue
                param_sq += param.detach().float().pow(2).sum().item()
                if param.grad is not None:
                    has_grad = True
                    grad_sq += param.grad.detach().float().pow(2).sum().item()
        return {
            "param_norm": math.sqrt(param_sq),
            "grad_norm": math.sqrt(grad_sq) if has_grad else None,
        }

    def all_reduce_network(self, accelerator, network):
        super().all_reduce_network(accelerator, network)
        unet = getattr(self, "_current_unet_for_logging", None)
        if unet is not None:
            self._last_ip_adapter_grad_norm = self.get_ip_adapter_norms(unet)["grad_norm"]

    def generate_step_logs(
        self,
        args,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        mean_grad_norm=None,
        mean_combined_norm=None,
    ):
        logs = super().generate_step_logs(
            args,
            current_loss,
            avr_loss,
            lr_scheduler,
            lr_descriptions,
            optimizer,
            keys_scaled,
            mean_norm,
            maximum_norm,
            mean_grad_norm,
            mean_combined_norm,
        )
        unet = getattr(self, "_current_unet_for_logging", None)
        if getattr(args, "anima_ip_adapter", False) and unet is not None:
            norms = self.get_ip_adapter_norms(unet)
            logs["anima_ip_adapter/param_norm"] = norms["param_norm"]
            grad_norm = getattr(self, "_last_ip_adapter_grad_norm", norms["grad_norm"])
            if grad_norm is not None:
                logs["anima_ip_adapter/grad_norm"] = grad_norm
        return logs

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        return noise_scheduler

    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=True):
        if is_train and getattr(args, "anima_train_ip_adapter", False):
            self.set_ip_adapter_trainable(accelerator.unwrap_model(unet), True)
        return super().on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train)

    def encode_images_to_latents(self, args, vae, images):
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage
        return vae.encode_pixels_to_latents(images)  # Keep 4D for input/output

    def shift_scale_latents(self, args, latents):
        # Latents already normalized by vae.encode with scale
        return latents

    def preprocess_reference_image(self, image, args, anima: anima_models.Anima):
        image = Image.fromarray(image[:, :, :3]).convert("RGB")
        max_area = getattr(args, "anima_reference_max_area", 1024 * 1024)
        if max_area is not None and max_area > 0 and image.width * image.height > max_area:
            scale = math.sqrt(max_area / (image.width * image.height))
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)

        multiple_of = qwen_image_autoencoder_kl.SCALE_FACTOR * anima.patch_spatial
        width = (image.width // multiple_of) * multiple_of
        height = (image.height // multiple_of) * multiple_of
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Reference image is too small after alignment: {image.width}x{image.height}. "
                f"Both dimensions must be at least {multiple_of}px."
            )

        left = (image.width - width) // 2
        top = (image.height - height) // 2
        image = image.crop((left, top, left + width, top + height))
        return image

    def load_reference_latents_from_batch(self, args, vae, batch, accelerator, weight_dtype, anima: anima_models.Anima):
        needs_vae_refs = getattr(args, "anima_multi_image_edit", False) or (
            getattr(args, "anima_ip_adapter", False) and args.anima_ip_adapter_feature_backend == "vae"
        )
        if not needs_vae_refs:
            return None

        reference_latents = batch.get("reference_latents", None)
        if reference_latents is not None:
            return reference_latents

        reference_image_paths = batch.get("reference_image_paths", None)
        if (
            (reference_image_paths is None or not any(len(paths or []) > 0 for paths in reference_image_paths))
            and getattr(args, "anima_self_reference_test", False)
            and batch.get("image_paths", None) is not None
        ):
            reference_image_paths = [[path] for path in batch["image_paths"]]

        if reference_image_paths is None:
            return None

        image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage
        encoded_batch = []
        for sample_paths in reference_image_paths:
            sample_latents = []
            for path in sample_paths or []:
                if not path:
                    continue
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Reference image not found: {path}")
                image = self.preprocess_reference_image(train_util.load_image(path), args, anima)
                image_tensor = image_transform(image).unsqueeze(0).to(accelerator.device, dtype=vae.dtype)
                with torch.no_grad():
                    latent = vae.encode_pixels_to_latents(image_tensor).to(accelerator.device, dtype=weight_dtype)
                if latent.ndim == 5:
                    latent = latent.squeeze(2)
                sample_latents.append(latent)
            encoded_batch.append(sample_latents)

        if not any(len(refs) > 0 for refs in encoded_batch):
            return None
        return encoded_batch

    def load_ip_adapter_embeds_from_batch(self, args, batch, accelerator, weight_dtype):
        if not getattr(args, "anima_ip_adapter", False):
            return None
        if args.anima_ip_adapter_feature_backend == "vae":
            return None

        reference_image_paths = batch.get("reference_image_paths", None)
        if (
            (reference_image_paths is None or not any(len(paths or []) > 0 for paths in reference_image_paths))
            and getattr(args, "anima_self_reference_test", False)
            and batch.get("image_paths", None) is not None
        ):
            reference_image_paths = [[path] for path in batch["image_paths"]]

        if reference_image_paths is None or not any(len(paths or []) > 0 for paths in reference_image_paths):
            return None

        if self.ip_adapter_feature_extractor is None:
            raise ValueError("IP-Adapter feature extractor was not initialized.")

        sample_features = []
        for sample_paths in reference_image_paths:
            paths = [path for path in (sample_paths or []) if path]
            if paths:
                features = self.ip_adapter_feature_extractor.extract(paths).to(accelerator.device, dtype=weight_dtype)
            else:
                features = torch.zeros((0, self.ip_adapter_feature_extractor.feature_dim), device=accelerator.device, dtype=weight_dtype)
            sample_features.append(features)

        if not any(features.shape[0] > 0 for features in sample_features):
            return None
        return sample_features

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        anima: anima_models.Anima = unet
        self._current_unet_for_logging = accelerator.unwrap_model(unet)

        # Sample noise
        if latents.ndim == 5:  # Fallback for 5D latents (old cache)
            latents = latents.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]
        noise = torch.randn_like(latents)

        # Get noisy model input and timesteps
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )
        timesteps = timesteps / 1000.0  # scale to [0, 1] range. timesteps is float32

        # Gradient checkpointing support
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)

        # Unpack text encoder conditions
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_conds[
            :4
        ]  # ignore caption_dropout_rate which is not needed for training step

        # Move to device
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        attn_mask = attn_mask.to(accelerator.device)
        t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
        t5_attn_mask = t5_attn_mask.to(accelerator.device)

        # Create padding mask
        bs = latents.shape[0]
        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=weight_dtype, device=accelerator.device)

        # Call model
        noisy_model_input = noisy_model_input.unsqueeze(2)  # 4D to 5D, [B, C, H, W] -> [B, C, 1, H, W]
        reference_latents = self.load_reference_latents_from_batch(args, self._current_vae, batch, accelerator, weight_dtype, anima)
        ip_adapter_embeds = self.load_ip_adapter_embeds_from_batch(args, batch, accelerator, weight_dtype)
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = anima(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                padding_mask=padding_mask,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=attn_mask,
                reference_latents=reference_latents,
                reference_t_offset_scale=args.anima_reference_t_offset_scale,
                ip_adapter_latents=None if ip_adapter_embeds is not None else reference_latents,
                ip_adapter_embeds=ip_adapter_embeds,
                use_ip_adapter=args.anima_ip_adapter,
                use_reference_sequence=args.anima_use_reference_sequence,
            )
        model_pred = model_pred.squeeze(2)  # 5D to 4D, [B, C, 1, H, W] -> [B, C, H, W]

        # Rectified flow target: noise - latents
        target = noise - latents

        # Loss weighting
        weighting = anima_train_utils.compute_loss_weighting_for_anima(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

        return model_pred, target, timesteps, weighting

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy,
        tokenize_strategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
    ) -> torch.Tensor:
        """Override base process_batch for caption dropout with cached text encoder outputs."""

        # Text encoder conditions
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        anima_text_encoding_strategy: strategy_anima.AnimaTextEncodingStrategy = text_encoding_strategy
        if text_encoder_outputs_list is not None:
            caption_dropout_rates = text_encoder_outputs_list[-1]
            text_encoder_outputs_list = text_encoder_outputs_list[:-1]

            # Apply caption dropout to cached outputs
            text_encoder_outputs_list = anima_text_encoding_strategy.drop_cached_text_encoder_outputs(
                *text_encoder_outputs_list, caption_dropout_rates=caption_dropout_rates
            )
            # Add the caption dropout rates back to the list for validation dataset (which is re-used batch items)
            batch["text_encoder_outputs_list"] = text_encoder_outputs_list + [caption_dropout_rates]

        self._current_vae = vae
        return super().process_batch(
            batch,
            text_encoders,
            unet,
            network,
            vae,
            noise_scheduler,
            vae_dtype,
            weight_dtype,
            accelerator,
            args,
            text_encoding_strategy,
            tokenize_strategy,
            is_train,
            train_text_encoder,
            train_unet,
        )

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec_dataclass(None, args, False, True, False, anima="preview").to_metadata_dict()

    def update_metadata(self, metadata, args):
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # Set first parameter's requires_grad to True to workaround Accelerate gradient checkpointing bug
        first_param = next(text_encoder.parameters())
        first_param.requires_grad_(True)

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        if getattr(args, "anima_train_ip_adapter", False):
            self.set_ip_adapter_trainable(unet, True)

        # The base NetworkTrainer only calls enable_gradient_checkpointing(cpu_offload=True/False),
        # so we re-apply with unsloth_offload if needed (after base has already enabled it).
        if self._use_unsloth_offload_checkpointing and args.gradient_checkpointing:
            unet.enable_gradient_checkpointing(unsloth_offload=True)

        if not self.is_swapping_blocks:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)

        model = unet
        model = accelerator.prepare(model, device_placement=[not self.is_swapping_blocks])
        accelerator.unwrap_model(model).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(model).prepare_block_swap_before_forward()

        return model

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            # prepare for next forward: because backward pass is not called, we need to prepare it here
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    # parser.add_argument("--fp8_scaled", action="store_true", help="Use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing or --blocks_to_swap.",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    trainer = AnimaNetworkTrainer()
    trainer.train(args)
