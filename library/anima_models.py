# Anima Model Architecture
# Original code: NVIDIA CORPORATION & AFFILIATES, licensed under Apache-2.0

import math
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as torch_checkpoint

from library import custom_offloading_utils, attention


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_device(elem, device) for elem in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        return x


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, (list, tuple)):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x


# Unsloth Offloaded Gradient Checkpointing
# Based on Unsloth Zoo by Daniel Han-Chen & the Unsloth team
try:
    from deepspeed.runtime.activation_checkpointing.checkpointing import detach_variable
except ImportError:

    def detach_variable(inputs, device=None):
        """Detach tensors from computation graph, optionally moving to a device.

        Reimplementation of deepspeed.runtime.activation_checkpointing.checkpointing.detach_variable
        for environments without DeepSpeed installed.
        """
        if isinstance(inputs, tuple):
            out = []
            for inp in inputs:
                if not isinstance(inp, torch.Tensor):
                    out.append(inp)
                    continue
                requires_grad = inp.requires_grad
                if device is not None:
                    x = inp.to(device=device)
                else:
                    x = inp
                x = x.detach()
                x.requires_grad = requires_grad
                out.append(x)
            return tuple(out)
        else:
            raise RuntimeError(
                "Only tuple of tensors is supported. Got Unsupported input type: ",
                type(inputs).__name__,
            )


class UnslothOffloadedGradientCheckpointer(torch.autograd.Function):
    """Saves VRAM by offloading activations to CPU RAM using non-blocking transfers.

    Compared to standard cpu_offload_checkpointing which uses blocking transfers,
    this uses non_blocking=True to hide CPU<->GPU transfer latency behind compute.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, forward_function, hidden_states, *args):
        # Remember the original device for backward pass (multi-GPU support)
        ctx.input_device = hidden_states.device
        saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        # NOTE: args stored directly on ctx (not via save_for_backward) because
        # the training loop holds references to these tensors, preventing GC.
        # Using save_for_backward for all args would add complexity for no benefit.
        ctx.args = args
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grads):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to(ctx.input_device, non_blocking=True).detach()
        hidden_states.requires_grad_(True)
        args = detach_variable(ctx.args)
        inputs = (hidden_states,) + args
        with torch.enable_grad():
            outputs = ctx.forward_function(*inputs)

        output_tensors = []
        grad_tensors = []
        for out, grad in zip(
            outputs if isinstance(outputs, tuple) else (outputs,), grads if isinstance(grads, tuple) else (grads,)
        ):
            if isinstance(out, torch.Tensor) and out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)
        torch.autograd.backward(output_tensors, grad_tensors)
        return (None,) + tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in inputs)


@torch._disable_dynamo
def unsloth_checkpoint(function, *args):
    """Wrapper for UnslothOffloadedGradientCheckpointer."""
    return UnslothOffloadedGradientCheckpointer.apply(function, *args)


from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# Utility functions: RoPE for DiT
def _rotate_half(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if not interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_base(
    t: torch.Tensor,
    freqs: torch.Tensor,
    start_positions: torch.Tensor = None,
    tensor_format: str = "sbhd",
    interleaved: bool = False,
) -> torch.Tensor:
    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

    if start_positions is not None:
        max_offset = torch.max(start_positions)
        assert max_offset + cur_seq_len <= max_seq_len, f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
        freqs = torch.concatenate([freqs[i : i + cur_seq_len] for i in start_positions], dim=1)

    assert cur_seq_len <= max_seq_len, f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    freqs = freqs[:cur_seq_len]

    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * cos_) + (_rotate_half(t, interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    start_positions: Union[torch.Tensor, None] = None,
    interleaved: bool = False,
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
    cp_size: int = 1,
) -> torch.Tensor:
    assert not (cp_size > 1 and start_positions is not None), "start_positions != None with CP SIZE > 1 is not supported!"

    assert tensor_format != "thd" or cu_seqlens is not None, "cu_seqlens must not be None when tensor_format is 'thd'."

    assert fused == False

    if tensor_format == "thd":
        cu_seqlens = cu_seqlens // cp_size
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        return torch.cat(
            [
                _apply_rotary_pos_emb_base(
                    x.unsqueeze(1),
                    freqs,
                    start_positions=(start_positions[idx : idx + 1] if start_positions is not None else None),
                    interleaved=interleaved,
                )
                for idx, x in enumerate(torch.split(t, seqlens))
            ]
        ).squeeze(1)

    if tensor_format == "sbhd":
        seqlen = t.size(0)
    elif tensor_format == "bshd":
        seqlen = t.size(1)
    else:
        raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
    return _apply_rotary_pos_emb_base(
        t,
        freqs,
        start_positions,
        tensor_format,
        interleaved=interleaved,
    )


# Basic building blocks
class RMSNorm(torch.nn.Module):
    """RMS Normalization for DiT blocks."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            output = self._norm(x.float()).type_as(x)
            return output * self.weight


class RMSNormNoAffine(torch.nn.Module):
    """RMS normalization without trainable affine parameters."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        del dim
        self.eps = eps

    def reset_parameters(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


class GPT2FeedForward(nn.Module):
    """GELU feedforward network."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._hidden_dim)
        if self._layer_id is not None:
            std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


# Attention module for DiT
class Attention(nn.Module):
    """Multi-head attention supporting both self-attention and cross-attention.

    Uses QK-norm (RMSNorm on q/k) and optional RoPE (only for self-attention).
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        qkv_format: str = "bshd",
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._query_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._context_dim)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)

        for layer in self.q_norm, self.k_norm, self.v_norm:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> tuple:
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        if self.is_selfattn and rope_emb is not None:
            q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=False)
            k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=False)

        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        attn_params: attention.AttentionParams,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        if q.dtype != v.dtype:
            if (not attn_params.supports_fp32 or attn_params.requires_same_dtype) and torch.is_autocast_enabled():
                # FlashAttention requires fp16/bf16, xformers require same dtype; only cast when autocast is active.
                target_dtype = v.dtype  # v has fp16/bf16 dtype
                q = q.to(target_dtype)
                k = k.to(target_dtype)
        # return self.compute_attention(q, k, v)
        qkv = [q, k, v]
        del q, k, v
        result = attention.attention(qkv, attn_params=attn_params)
        return self.output_dropout(self.output_proj(result))


class AnimaVisualConditionAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_size: int,
        num_heads: int,
        num_feature_tokens: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        linear_adapter: bool = False,
        mlp_adapter: bool = False,
        norm_linear_adapter: bool = False,
        omni_adapter: bool = False,
    ):
        super().__init__()
        if sum(bool(v) for v in (linear_adapter, mlp_adapter, norm_linear_adapter, omni_adapter)) > 1:
            raise ValueError("Only one of linear_adapter, mlp_adapter, norm_linear_adapter, and omni_adapter can be enabled.")
        if hidden_size % num_heads != 0:
            raise ValueError(f"Visual condition hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_feature_tokens = num_feature_tokens
        self.linear_adapter = linear_adapter
        self.mlp_adapter = mlp_adapter
        self.norm_linear_adapter = norm_linear_adapter
        self.omni_adapter = omni_adapter
        if self.linear_adapter:
            self.feature_proj = nn.Linear(feature_dim, hidden_size * num_feature_tokens, bias=True)
            self.out_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        elif self.mlp_adapter:
            self.feature_proj = nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2, bias=True),
                nn.GELU(),
                nn.Linear(feature_dim * 2, hidden_size * num_feature_tokens, bias=True),
            )
            self.out_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        elif self.norm_linear_adapter:
            self.feature_norm = RMSNormNoAffine(feature_dim, eps=1e-6)
            self.feature_proj = nn.Linear(feature_dim, hidden_size, bias=False)
            self.feature_expand = nn.Linear(feature_dim, hidden_size * num_feature_tokens, bias=False)
        elif self.omni_adapter:
            self.feature_norm = RMSNorm(feature_dim, eps=1e-5)
            self.feature_proj = nn.Linear(feature_dim, hidden_size, bias=True)
            self.feature_expand = nn.Linear(feature_dim, hidden_size * num_feature_tokens, bias=True)
            self.rotary_emb = AdapterRotaryEmbedding(hidden_size // num_heads)
            self.image_rotary_emb = AdapterImageRotaryEmbedding(hidden_size // num_heads, rope_theta=256.0)
            self.refiner = nn.ModuleList(
                [
                    OmniRefinerBlock(
                        dim=hidden_size,
                        num_heads=num_heads,
                        mlp_ratio=8.0 / 3.0,
                        norm_eps=1e-5,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.out_norm = nn.Identity()
        else:
            self.feature_norm = RMSNorm(feature_dim, eps=1e-6)
            self.source_proj = nn.Linear(feature_dim, hidden_size, bias=True)
            self.visual_queries = nn.Parameter(torch.empty(num_feature_tokens, hidden_size))
            self.rotary_emb = AdapterRotaryEmbedding(hidden_size // num_heads)
            self.refiner = nn.ModuleList(
                [
                    LLMAdapterTransformerBlock(
                        source_dim=hidden_size,
                        model_dim=hidden_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        self_attn=True,
                        layer_norm=False,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.out_norm = LLMAdapterRMSNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.feature_dim)
        if self.linear_adapter:
            torch.nn.init.trunc_normal_(self.feature_proj.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.feature_proj.bias)
            self.out_norm.reset_parameters()
        elif self.mlp_adapter:
            torch.nn.init.trunc_normal_(self.feature_proj[0].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.feature_proj[0].bias)
            hidden_std = 1.0 / math.sqrt(self.feature_dim * 2)
            torch.nn.init.trunc_normal_(self.feature_proj[2].weight, std=hidden_std, a=-3 * hidden_std, b=3 * hidden_std)
            torch.nn.init.zeros_(self.feature_proj[2].bias)
            self.out_norm.reset_parameters()
        elif self.norm_linear_adapter:
            self.feature_norm.reset_parameters()
            torch.nn.init.trunc_normal_(self.feature_proj.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.feature_expand.weight, std=std, a=-3 * std, b=3 * std)
        elif self.omni_adapter:
            self.feature_norm.reset_parameters()
            torch.nn.init.trunc_normal_(self.feature_proj.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.feature_expand.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.feature_proj.bias)
            torch.nn.init.zeros_(self.feature_expand.bias)
            for layer in self.refiner:
                layer.init_weights()
            torch.nn.init.zeros_(self.out_proj.weight)
        else:
            self.feature_norm.reset_parameters()
            torch.nn.init.trunc_normal_(self.source_proj.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.source_proj.bias)
            query_std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.visual_queries, std=query_std, a=-3 * query_std, b=3 * query_std)
            for layer in self.refiner:
                layer.init_weights()
            self.out_norm.weight.data.fill_(1.0)

    def forward(
        self,
        features: torch.Tensor,
        attn_params: attention.AttentionParams,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del attn_params, rope_emb
        if features.ndim == 2:
            features = features.unsqueeze(1)
        elif features.ndim == 4:
            batch, height, width, dim = features.shape
            features = features.reshape(batch, height * width, dim)
        elif features.ndim == 5:
            batch, num_refs, height, width, dim = features.shape
            features = features.reshape(batch, num_refs * height * width, dim)

        batch, num_features, _ = features.shape
        if num_features == 0:
            return features.new_zeros((batch, 0, self.hidden_size))

        if self.linear_adapter or self.mlp_adapter:
            pooled = features.mean(dim=1)
            if self.mlp_adapter:
                proj_weight = self.feature_proj[0].weight
                tokens = self.feature_proj(pooled.to(proj_weight.dtype))
            else:
                tokens = self.feature_proj(pooled.to(self.feature_proj.weight.dtype))
            tokens = tokens.reshape(batch, self.num_feature_tokens, self.hidden_size)
            return self.out_norm(tokens)

        if self.norm_linear_adapter:
            normed = self.feature_norm(features).to(self.feature_proj.weight.dtype)
            if num_features == 1:
                tokens = self.feature_expand(normed[:, 0])
                return tokens.reshape(batch, self.num_feature_tokens, self.hidden_size)
            return self.feature_proj(normed)

        if self.omni_adapter:
            normed = self.feature_norm(features).to(self.feature_proj.weight.dtype)
            if num_features == 1:
                tokens = self.feature_expand(normed[:, 0]).reshape(batch, self.num_feature_tokens, self.hidden_size)
            else:
                tokens = self.feature_proj(normed)
            position_embeddings = self._get_omni_position_embeddings(tokens)
            for layer in self.refiner:
                tokens = layer(tokens, position_embeddings=position_embeddings)
            return self.out_proj(self.out_norm(tokens))

        source = self.source_proj(self.feature_norm(features))
        tokens = self.visual_queries.unsqueeze(0).expand(batch, -1, -1).to(dtype=source.dtype, device=source.device)
        position_ids = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        position_ids_source = torch.arange(source.shape[1], device=source.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(tokens, position_ids)
        position_embeddings_source = self.rotary_emb(source, position_ids_source)
        for layer in self.refiner:
            tokens = layer(
                tokens,
                source,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings_source,
        )
        return self.out_norm(tokens)

    def _get_omni_position_embeddings(self, tokens: torch.Tensor):
        seq_len = tokens.shape[1]
        base_grid = int(math.isqrt(seq_len))
        if base_grid * base_grid == seq_len:
            return self.image_rotary_emb(tokens, (base_grid, base_grid), num_images=1)

        # Known grid sizes for different feature backends. Preserve per-image 2D
        # positions when multiple reference images are concatenated.
        for grid_size, grid_name in [(12, "CCIP"), (16, "SigLIP2")]:
            tokens_per_image = grid_size * grid_size
            if seq_len % tokens_per_image == 0:
                return self.image_rotary_emb(tokens, (grid_size, grid_size), num_images=seq_len // tokens_per_image)

        position_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        return self.rotary_emb(tokens, position_ids)


# Positional Embeddings
class VideoPositionEmb(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    def seq_dim(self) -> int:
        return 1

    def forward(self, x_B_T_H_W_C: torch.Tensor, fps: Optional[torch.Tensor]) -> torch.Tensor:
        B_T_H_W_C = x_B_T_H_W_C.shape
        embeddings = self.generate_embeddings(B_T_H_W_C, fps=fps)
        return embeddings

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps: Optional[torch.Tensor]) -> Any:
        raise NotImplementedError


class VideoRopePosition3DEmb(VideoPositionEmb):
    """3D Rotary Position Embedding for video (T, H, W) dimensions."""

    def __init__(
        self,
        *,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.enable_fps_modulation = enable_fps_modulation
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().to(self.dim_spatial_range.device)
        self.dim_spatial_range = torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(self.dim_spatial_range.device) / dim_h
        self.dim_temporal_range = torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(self.dim_spatial_range.device) / dim_t

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ) -> torch.Tensor:
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        B, T, H, W, _ = B_T_H_W_C
        assert (
            H <= self.max_h and W <= self.max_w
        ), f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        if self.enable_fps_modulation:
            uniform_fps = (fps is None) or (fps.min() == fps.max())
            assert (
                uniform_fps or B == 1 or T == 1
            ), "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"

            if fps is None:
                assert T == 1, "T should be 1 for image batch."
                half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
            else:
                half_emb_t = torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    @property
    def seq_dim(self) -> int:
        return 0

    def generate_embeddings_from_ids(
        self,
        ids_L_3: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ) -> torch.Tensor:
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        ids_L_3 = ids_L_3.to(device=self.dim_spatial_range.device, dtype=torch.float32)
        t = ids_L_3[:, 0]
        h = ids_L_3[:, 1]
        w = ids_L_3[:, 2]

        if self.enable_fps_modulation and fps is not None:
            t = t / fps[:1].to(device=t.device, dtype=t.dtype) * self.base_fps

        half_emb_t = t[:, None] * temporal_freqs[None]
        half_emb_h = h[:, None] * h_spatial_freqs[None]
        half_emb_w = w[:, None] * w_spatial_freqs[None]
        emb_L_D = torch.cat([half_emb_t, half_emb_h, half_emb_w] * 2, dim=-1)
        return emb_L_D[:, None, None, :].float()


class LearnablePosEmbAxis(VideoPositionEmb):
    """Learnable axis-decomposed positional embeddings."""

    def __init__(
        self,
        *,
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        self.interpolation = interpolation
        assert self.interpolation in ["crop"], f"Unknown interpolation method {self.interpolation}"
        self.model_channels = model_channels

        self.pos_emb_h = nn.Parameter(torch.zeros(len_h, model_channels))
        self.pos_emb_w = nn.Parameter(torch.zeros(len_w, model_channels))
        self.pos_emb_t = nn.Parameter(torch.zeros(len_t, model_channels))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.model_channels)
        torch.nn.init.trunc_normal_(self.pos_emb_h, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_emb_w, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_emb_t, std=std, a=-3 * std, b=3 * std)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, H, W, _ = B_T_H_W_C
        if self.interpolation == "crop":
            emb_h_H = self.pos_emb_h[:H]
            emb_w_W = self.pos_emb_w[:W]
            emb_t_T = self.pos_emb_t[:T]
            emb = (
                repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W)
                + repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W)
                + repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H)
            )
            assert list(emb.shape)[:4] == [B, T, H, W], f"bad shape: {list(emb.shape)[:4]} != {B, T, H, W}"
        else:
            raise ValueError(f"Unknown interpolation method {self.interpolation}")

        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(1e-6, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return emb / norm.to(emb.dtype)


# Timestep Embedding
class Timesteps(nn.Module):
    """Sinusoidal timestep features."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2, f"Expected 2D input, got {timesteps_B_T.ndim}"
        in_dtype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return rearrange(emb.to(dtype=in_dtype), "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    """Projects timestep features to model dimension, with optional AdaLN-LoRA."""

    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.in_dim)
        torch.nn.init.trunc_normal_(self.linear_1.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self.out_dim)
        torch.nn.init.trunc_normal_(self.linear_2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, sample: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_T_3D = emb
            emb_B_T_D = sample
        else:
            adaln_lora_B_T_3D = None
            emb_B_T_D = emb

        return emb_B_T_D, adaln_lora_B_T_3D


# Commented out Fourier Features (not used in Anima). Kept for reference.
# class FourierFeatures(nn.Module):
#     """Fourier feature transform: [B] -> [B, D]."""

#     def __init__(self, num_channels: int, bandwidth: int = 1, normalize: bool = False):
#         super().__init__()
#         self.register_buffer("freqs", 2 * np.pi * bandwidth * torch.randn(num_channels), persistent=True)
#         self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels), persistent=True)
#         self.gain = np.sqrt(2) if normalize else 1
#         self.bandwidth = bandwidth
#         self.num_channels = num_channels
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         generator = torch.Generator()
#         generator.manual_seed(0)
#         self.freqs = 2 * np.pi * self.bandwidth * torch.randn(self.num_channels, generator=generator).to(self.freqs.device)
#         self.phases = 2 * np.pi * torch.rand(self.num_channels, generator=generator).to(self.freqs.device)

#     def forward(self, x: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
#         in_dtype = x.dtype
#         x = x.to(torch.float32).ger(self.freqs.to(torch.float32)).add(self.phases.to(torch.float32))
#         x = x.cos().mul(self.gain * gain).to(in_dtype)
#         return x


# Patch Embedding
class PatchEmbed(nn.Module):
    """Patch embedding: (B, C, T, H, W) -> (B, T', H', W', D)"""

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False),
        )
        self.dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert (
            H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        ), f"H,W {(H, W)} should be divisible by spatial_patch_size {self.spatial_patch_size}"
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return x


# Final Layer
class FinalLayer(nn.Module):
    """Final layer with AdaLN modulation + unpatchify."""

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False))

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            torch.nn.init.trunc_normal_(self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)

        self.layer_norm.reset_parameters()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        use_fp32: bool = False,
    ):
        # Compute AdaLN modulation parameters (in float32 when fp16 to avoid overflow in Linear layers)
        with torch.autocast(device_type=x_B_T_H_W_D.device.type, dtype=torch.float32, enabled=use_fp32):
            if self.use_adaln_lora:
                assert adaln_lora_B_T_3D is not None
                shift_B_T_D, scale_B_T_D = (
                    self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
                ).chunk(2, dim=-1)
            else:
                shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

        shift_B_T_1_1_D = rearrange(shift_B_T_D, "b t d -> b t 1 1 d")
        scale_B_T_1_1_D = rearrange(scale_B_T_D, "b t d -> b t 1 1 d")

        x_B_T_H_W_D = self.layer_norm(x_B_T_H_W_D) * (1 + scale_B_T_1_1_D) + shift_B_T_1_1_D
        x_B_T_H_W_O = self.linear(x_B_T_H_W_D)
        return x_B_T_H_W_O


# Transformer Block (DiT Block)
class Block(nn.Module):
    """Transformer block with self-attention + cross-attention + MLP, each modulated by AdaLN.

    Each sublayer: x = x + gate * sublayer(norm(x) * (1 + scale) + shift)
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            x_dim,
            None,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
        )

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(
            x_dim,
            context_dim,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
        )

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False, unsloth_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload if not unsloth_offload else False
        self.unsloth_offload_checkpointing = unsloth_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    def enable_ip_adapter(
        self,
        scale: float = 1.0,
        feature_dim: Optional[int] = None,
        num_feature_tokens: int = 4,
        linear_adapter: bool = False,
        mlp_adapter: bool = False,
        norm_linear_adapter: bool = False,
        omni_adapter: bool = False,
    ) -> None:
        del feature_dim, num_feature_tokens
        del scale, linear_adapter, mlp_adapter, norm_linear_adapter, omni_adapter

    def set_ip_adapter_scale(self, scale: float) -> None:
        del scale

    def _forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params: attention.AttentionParams,
        use_fp32: bool = False,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_fp32:
            # Cast to float32 for better numerical stability in residual connections. Each module will cast back to float16 by enclosing autocast context.
            x_B_T_H_W_D = x_B_T_H_W_D.float()

        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        # Compute AdaLN modulation parameters (in float32 when fp16 to avoid overflow in Linear layers)
        with torch.autocast(device_type=x_B_T_H_W_D.device.type, dtype=torch.float32, enabled=use_fp32):
            if self.use_adaln_lora:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                    self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(
                    3, dim=-1
                )
            else:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                    emb_B_T_D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = self.adaln_modulation_cross_attn(
                    emb_B_T_D
                ).chunk(3, dim=-1)
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # Reshape for broadcasting: (B, T, D) -> (B, T, 1, 1, D)
        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _adaln_fn(_x, _norm_layer, _scale, _shift):
            return _norm_layer(_x) * (1 + _scale) + _shift

        # 1. Self-attention
        normalized_x = _adaln_fn(x_B_T_H_W_D, self.layer_norm_self_attn, scale_self_attn_B_T_1_1_D, shift_self_attn_B_T_1_1_D)
        result = rearrange(
            self.self_attn(
                rearrange(normalized_x, "b t h w d -> b (t h w) d"),
                attn_params,
                None,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result

        # 2. Cross-attention
        normalized_x = _adaln_fn(x_B_T_H_W_D, self.layer_norm_cross_attn, scale_cross_attn_B_T_1_1_D, shift_cross_attn_B_T_1_1_D)
        result = rearrange(
            self.cross_attn(
                rearrange(normalized_x, "b t h w d -> b (t h w) d"),
                attn_params,
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = result * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D

        # 3. MLP
        normalized_x = _adaln_fn(x_B_T_H_W_D, self.layer_norm_mlp, scale_mlp_B_T_1_1_D, shift_mlp_B_T_1_1_D)
        result = self.mlp(normalized_x)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result

        return x_B_T_H_W_D

    def _forward_flat(
        self,
        x_B_L_D: torch.Tensor,
        emb_B_1_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params: attention.AttentionParams,
        use_fp32: bool = False,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_1_3D: Optional[torch.Tensor] = None,
        ip_adapter_tokens: Optional[torch.Tensor] = None,
        ip_adapter_query_len: Optional[int] = None,
    ) -> torch.Tensor:
        if use_fp32:
            x_B_L_D = x_B_L_D.float()

        with torch.autocast(device_type=x_B_L_D.device.type, dtype=torch.float32, enabled=use_fp32):
            if self.use_adaln_lora:
                assert adaln_lora_B_1_3D is not None
                shift_self, scale_self, gate_self = (
                    self.adaln_modulation_self_attn(emb_B_1_D) + adaln_lora_B_1_3D
                ).chunk(3, dim=-1)
                shift_cross, scale_cross, gate_cross = (
                    self.adaln_modulation_cross_attn(emb_B_1_D) + adaln_lora_B_1_3D
                ).chunk(3, dim=-1)
                shift_mlp, scale_mlp, gate_mlp = (self.adaln_modulation_mlp(emb_B_1_D) + adaln_lora_B_1_3D).chunk(3, dim=-1)
            else:
                shift_self, scale_self, gate_self = self.adaln_modulation_self_attn(emb_B_1_D).chunk(3, dim=-1)
                shift_cross, scale_cross, gate_cross = self.adaln_modulation_cross_attn(emb_B_1_D).chunk(3, dim=-1)
                shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_1_D).chunk(3, dim=-1)

        def expand_param(x):
            return x.expand(-1, x_B_L_D.shape[1], -1)

        def adaln(_x, _norm, _scale, _shift):
            return _norm(_x) * (1 + expand_param(_scale)) + expand_param(_shift)

        normalized_x = adaln(x_B_L_D, self.layer_norm_self_attn, scale_self, shift_self)
        result = self.self_attn(normalized_x, attn_params, None, rope_emb=rope_emb_L_1_1_D)
        x_B_L_D = x_B_L_D + expand_param(gate_self) * result

        normalized_x = adaln(x_B_L_D, self.layer_norm_cross_attn, scale_cross, shift_cross)
        result = self.cross_attn(normalized_x, attn_params, crossattn_emb, rope_emb=rope_emb_L_1_1_D)
        x_B_L_D = x_B_L_D + expand_param(gate_cross) * result

        normalized_x = adaln(x_B_L_D, self.layer_norm_mlp, scale_mlp, shift_mlp)
        result = self.mlp(normalized_x)
        x_B_L_D = x_B_L_D + expand_param(gate_mlp) * result
        return x_B_L_D

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params: attention.AttentionParams,
        use_fp32: bool = False,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            if self.unsloth_offload_checkpointing:
                # Unsloth: async non-blocking CPU RAM offload (fastest offload method)
                return unsloth_checkpoint(
                    self._forward,
                    x_B_T_H_W_D,
                    emb_B_T_D,
                    crossattn_emb,
                    attn_params,
                    use_fp32,
                    rope_emb_L_1_1_D,
                    adaln_lora_B_T_3D,
                    extra_per_block_pos_emb,
                )
            elif self.cpu_offload_checkpointing:
                # Standard cpu offload: blocking transfers
                def create_custom_forward(func):
                    def custom_forward(*inputs):
                        # Determine original device from first tensor input
                        device = next(t.device for t in inputs if isinstance(t, torch.Tensor))
                        device_inputs = to_device(inputs, device)
                        outputs = func(*device_inputs)
                        return to_cpu(outputs)

                    return custom_forward

                return torch_checkpoint(
                    create_custom_forward(self._forward),
                    x_B_T_H_W_D,
                    emb_B_T_D,
                    crossattn_emb,
                    attn_params,
                    use_fp32,
                    rope_emb_L_1_1_D,
                    adaln_lora_B_T_3D,
                    extra_per_block_pos_emb,
                    use_reentrant=False,
                )
            else:
                # Standard gradient checkpointing (no offload)
                return torch_checkpoint(
                    self._forward,
                    x_B_T_H_W_D,
                    emb_B_T_D,
                    crossattn_emb,
                    attn_params,
                    use_fp32,
                    rope_emb_L_1_1_D,
                    adaln_lora_B_T_3D,
                    extra_per_block_pos_emb,
                    use_reentrant=False,
                )
        else:
            return self._forward(
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                attn_params,
                use_fp32,
                rope_emb_L_1_1_D,
                adaln_lora_B_T_3D,
                extra_per_block_pos_emb,
            )

    def forward_flat(
        self,
        x_B_L_D: torch.Tensor,
        emb_B_1_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params: attention.AttentionParams,
        use_fp32: bool = False,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_1_3D: Optional[torch.Tensor] = None,
        ip_adapter_tokens: Optional[torch.Tensor] = None,
        ip_adapter_query_len: Optional[int] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            if self.unsloth_offload_checkpointing:
                # Unsloth: async non-blocking CPU RAM offload (fastest offload method)
                return unsloth_checkpoint(
                    self._forward_flat,
                    x_B_L_D,
                    emb_B_1_D,
                    crossattn_emb,
                    attn_params,
                    use_fp32,
                    rope_emb_L_1_1_D,
                    adaln_lora_B_1_3D,
                    ip_adapter_tokens,
                    ip_adapter_query_len,
                )
            elif self.cpu_offload_checkpointing:
                # Standard cpu offload: blocking transfers
                def create_custom_forward(func):
                    def custom_forward(*inputs):
                        # Determine original device from first tensor input
                        device = next(t.device for t in inputs if isinstance(t, torch.Tensor))
                        device_inputs = to_device(inputs, device)
                        outputs = func(*device_inputs)
                        return to_cpu(outputs)

                    return custom_forward

                return torch_checkpoint(
                    create_custom_forward(self._forward_flat),
                    x_B_L_D,
                    emb_B_1_D,
                    crossattn_emb,
                    attn_params,
                    use_fp32,
                    rope_emb_L_1_1_D,
                    adaln_lora_B_1_3D,
                    ip_adapter_tokens,
                    ip_adapter_query_len,
                    use_reentrant=False,
                )
            else:
                # Standard gradient checkpointing (no offload)
                return torch_checkpoint(
                    self._forward_flat,
                    x_B_L_D,
                    emb_B_1_D,
                    crossattn_emb,
                    attn_params,
                    use_fp32,
                    rope_emb_L_1_1_D,
                    adaln_lora_B_1_3D,
                    ip_adapter_tokens,
                    ip_adapter_query_len,
                    use_reentrant=False,
                )

        return self._forward_flat(
            x_B_L_D,
            emb_B_1_D,
            crossattn_emb,
            attn_params,
            use_fp32,
            rope_emb_L_1_1_D,
            adaln_lora_B_1_3D,
            ip_adapter_tokens,
            ip_adapter_query_len,
        )


# Main DiT Model: MiniTrainDIT (renamed to Anima)
class Anima(nn.Module):
    """Cosmos-Predict2 DiT model for image/video generation.

    28 transformer blocks with AdaLN-LoRA modulation, 3D RoPE, and optional LLM Adapter.
    """

    LATENT_CHANNELS = 16

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        use_llm_adapter: bool = False,
        attn_mode: str = "torch",
        split_attn: bool = False,
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.crossattn_emb_channels = crossattn_emb_channels
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.use_llm_adapter = use_llm_adapter

        self.attn_mode = attn_mode
        self.split_attn = split_attn

        # Block swap support
        self.blocks_to_swap = None
        self.offloader: Optional[custom_offloading_utils.ModelOffloader] = None

        self.build_patch_embed()
        self.build_pos_embed()
        self.visual_condition_adapter: Optional[AnimaVisualConditionAdapter] = None
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )

        if self.use_llm_adapter:
            self.llm_adapter = LLMAdapter(
                source_dim=1024,
                target_dim=1024,
                model_dim=1024,
                num_layers=6,
                self_attn=True,
            )

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
        )

        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)
        self.init_weights()

    def init_weights(self) -> None:
        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.reset_parameters()
        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()
        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

    def enable_gradient_checkpointing(self, cpu_offload: bool = False, unsloth_offload: bool = False):
        for block in self.blocks:
            block.enable_gradient_checkpointing(cpu_offload=cpu_offload, unsloth_offload=unsloth_offload)

    def disable_gradient_checkpointing(self):
        for block in self.blocks:
            block.disable_gradient_checkpointing()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def build_patch_embed(self) -> None:
        in_channels = self.in_channels + 1 if self.concat_padding_mask else self.in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=in_channels,
            out_channels=self.model_channels,
        )

    def build_pos_embed(self) -> None:
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
        )
        self.pos_embedder = cls_type(**kwargs)

        if self.extra_per_block_abs_pos_emb:
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            self.extra_pos_embedder = LearnablePosEmbAxis(**kwargs)

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        from torchvision import transforms

        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1)
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)

        return x_B_T_H_W_D, None, extra_pos_emb

    def _prepare_flat_tokens(
        self,
        x_B_C_T_H_W: torch.Tensor,
        t_offset: int = 0,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
        from torchvision import transforms

        if self.concat_padding_mask:
            if padding_mask is None:
                padding_mask = torch.zeros(
                    x_B_C_T_H_W.shape[0],
                    1,
                    x_B_C_T_H_W.shape[-2],
                    x_B_C_T_H_W.shape[-1],
                    dtype=x_B_C_T_H_W.dtype,
                    device=x_B_C_T_H_W.device,
                )
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1)

        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        B, T, H, W, D = x_B_T_H_W_D.shape
        t = torch.arange(t_offset, t_offset + T, device=x_B_T_H_W_D.device)
        h = torch.arange(H, device=x_B_T_H_W_D.device)
        w = torch.arange(W, device=x_B_T_H_W_D.device)
        ids = torch.cartesian_prod(t, h, w)
        return rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d"), ids, (T, H, W)

    def enable_ip_adapter(
        self,
        scale: float = 1.0,
        feature_dim: Optional[int] = None,
        num_feature_tokens: int = 4,
        linear_adapter: bool = False,
        mlp_adapter: bool = False,
        norm_linear_adapter: bool = False,
        omni_adapter: bool = False,
    ) -> None:
        if feature_dim is None:
            return
        if self.visual_condition_adapter is None:
            self.visual_condition_adapter = AnimaVisualConditionAdapter(
                feature_dim=feature_dim,
                hidden_size=self.crossattn_emb_channels,
                num_heads=self.num_heads,
                num_feature_tokens=num_feature_tokens,
                linear_adapter=linear_adapter,
                mlp_adapter=mlp_adapter,
                norm_linear_adapter=norm_linear_adapter,
                omni_adapter=omni_adapter,
            )

    def project_ip_adapter_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.visual_condition_adapter is None:
            raise ValueError("IP-Adapter is not enabled.")
        attn_params = attention.AttentionParams.create_attention_params(self.attn_mode, self.split_attn)
        return self.visual_condition_adapter(features, attn_params)

    def set_ip_adapter_scale(self, scale: float) -> None:
        del scale

    def _prepare_reference_flat_tokens(
        self,
        reference_latents: list[list[torch.Tensor]],
        dtype: torch.dtype,
        device: torch.device,
        t_offset_scale: int = 10,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if reference_latents is None or len(reference_latents) == 0:
            return None, None, None

        batch_tokens = []
        batch_ids = []
        seq_lens = []
        for refs in reference_latents:
            sample_tokens = []
            sample_ids = []
            for ref_index, ref_latent in enumerate(refs or []):
                if ref_latent.ndim == 4:
                    ref_latent = ref_latent.unsqueeze(2)
                ref_latent = ref_latent.to(device=device, dtype=dtype)
                tokens, ids, _ = self._prepare_flat_tokens(ref_latent, t_offset=t_offset_scale * (ref_index + 1), padding_mask=None)
                sample_tokens.append(tokens.squeeze(0))
                sample_ids.append(ids)
            if sample_tokens:
                sample_tokens = torch.cat(sample_tokens, dim=0)
                sample_ids = torch.cat(sample_ids, dim=0)
            else:
                sample_tokens = torch.zeros((0, self.model_channels), dtype=dtype, device=device)
                sample_ids = torch.zeros((0, 3), dtype=torch.long, device=device)
            batch_tokens.append(sample_tokens)
            batch_ids.append(sample_ids)
            seq_lens.append(sample_tokens.shape[0])

        max_len = max(seq_lens)
        if max_len == 0:
            return None, None, None

        padded_tokens = []
        padded_ids = []
        attention_mask = []
        for tokens, ids, seq_len in zip(batch_tokens, batch_ids, seq_lens):
            pad_len = max_len - seq_len
            if pad_len > 0:
                tokens = F.pad(tokens, (0, 0, 0, pad_len))
                ids = F.pad(ids, (0, 0, 0, pad_len))
            padded_tokens.append(tokens)
            padded_ids.append(ids)
            attention_mask.append(
                torch.cat(
                    [
                        torch.ones(seq_len, dtype=torch.bool, device=device),
                        torch.zeros(pad_len, dtype=torch.bool, device=device),
                    ]
                )
            )

        return torch.stack(padded_tokens), torch.stack(padded_ids), torch.stack(attention_mask)

    def unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        x_B_C_Tt_Hp_Wp = rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )
        return x_B_C_Tt_Hp_Wp

    def enable_block_swap(self, num_blocks: int, device: torch.device):
        self.blocks_to_swap = num_blocks

        assert (
            self.blocks_to_swap <= self.num_blocks - 2
        ), f"Cannot swap more than {self.num_blocks - 2} blocks. Requested: {self.blocks_to_swap} blocks."

        self.offloader = custom_offloading_utils.ModelOffloader(self.blocks, self.blocks_to_swap, device)
        logger.info(f"Anima: Block swap enabled. Swapping {num_blocks} blocks, total blocks: {self.num_blocks}, device: {device}.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # Move all modules to device except blocks (which are managed by offloader)
        if self.blocks_to_swap:
            save_blocks = self.blocks
            self.blocks = None  # Use None to skip .to() on blocks (consistent with flux_models.py)

        self.to(device)

        if self.blocks_to_swap:
            self.blocks = save_blocks

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.set_forward_only(True)
        self.prepare_block_swap_before_forward()
        print(f"Anima: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.set_forward_only(False)
        self.prepare_block_swap_before_forward()
        print(f"Anima: Block swap set to forward and backward.")

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def forward_mini_train_dit(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
        t5_input_ids: Optional[torch.Tensor] = None,
        t5_attn_mask: Optional[torch.Tensor] = None,
        reference_latents: Optional[list[list[torch.Tensor]]] = None,
        reference_t_offset_scale: int = 10,
        ip_adapter_latents: Optional[list[list[torch.Tensor]]] = None,
        ip_adapter_embeds: Optional[torch.Tensor] = None,
        use_ip_adapter: bool = False,
        use_reference_sequence: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x_B_C_T_H_W: (B, C, T, H, W) noisy latents
            timesteps_B_T: (B,) or (B, T) timesteps
            crossattn_emb: (B, N, D) cross-attention embeddings (or raw Qwen3 prompt_embeds if t5_input_ids provided)
            fps: Optional frames per second
            padding_mask: Optional padding mask
            source_attention_mask: Optional attention mask for Qwen3 embeddings (used with LLM adapter)
            t5_input_ids: Optional T5 token IDs (triggers LLM adapter when provided)
            t5_attn_mask: Optional T5 attention mask
        """
        # Run LLM adapter inside forward for correct DDP gradient synchronization
        if t5_input_ids is not None and self.use_llm_adapter and hasattr(self, "llm_adapter"):
            crossattn_emb = self.llm_adapter(
                source_hidden_states=crossattn_emb,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=source_attention_mask,
            )
            if t5_attn_mask is not None:
                crossattn_emb[~t5_attn_mask.bool()] = 0

        if use_ip_adapter and ip_adapter_latents is None:
            ip_adapter_latents = reference_latents

        has_reference_latents = (
            use_reference_sequence and reference_latents is not None and any(len(refs or []) > 0 for refs in reference_latents)
        )
        has_ip_adapter_latents = ip_adapter_latents is not None and any(len(refs or []) > 0 for refs in ip_adapter_latents)
        has_ip_adapter_embeds = ip_adapter_embeds is not None and (
            any(embed.shape[0] > 0 for embed in ip_adapter_embeds)
            if isinstance(ip_adapter_embeds, list)
            else ip_adapter_embeds.shape[1] > 0
        )

        if has_reference_latents or (use_ip_adapter and (has_ip_adapter_latents or has_ip_adapter_embeds)):
            if self.extra_per_block_abs_pos_emb:
                raise NotImplementedError("Multi-image reference conditioning does not support extra_per_block_abs_pos_emb yet.")
            if x_B_C_T_H_W.shape[2] != 1:
                raise NotImplementedError("Multi-image reference/IP-Adapter conditioning currently supports image training (T=1) only.")
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)

            outputs = []
            attn_params = attention.AttentionParams.create_attention_params(self.attn_mode, self.split_attn)
            for batch_index in range(x_B_C_T_H_W.shape[0]):
                x_i = x_B_C_T_H_W[batch_index : batch_index + 1]
                padding_i = padding_mask[batch_index : batch_index + 1] if padding_mask is not None else None
                target_tokens, target_ids, target_grid = self._prepare_flat_tokens(x_i, t_offset=0, padding_mask=padding_i)

                ref_tokens = []
                ref_ids = []
                sequence_reference_latents = (reference_latents[batch_index] or []) if use_reference_sequence else []
                for ref_index, ref_latent in enumerate(sequence_reference_latents):
                    if ref_latent.ndim == 4:
                        ref_latent = ref_latent.unsqueeze(2)
                    ref_latent = ref_latent.to(device=x_i.device, dtype=x_i.dtype)
                    tokens, ids, _ = self._prepare_flat_tokens(
                        ref_latent,
                        t_offset=reference_t_offset_scale * (ref_index + 1),
                        padding_mask=None,
                    )
                    ref_tokens.append(tokens)
                    ref_ids.append(ids)

                visual_context_tokens = []
                ip_tokens = None
                visual_latent_tokens = []
                visual_latent_ids = []
                if ip_adapter_embeds is not None:
                    if isinstance(ip_adapter_embeds, list):
                        sample_embeds = ip_adapter_embeds[batch_index].unsqueeze(0)
                    else:
                        sample_embeds = ip_adapter_embeds[batch_index : batch_index + 1]
                    tokens = self.project_ip_adapter_features(sample_embeds.to(x_i.device, x_i.dtype))
                    visual_context_tokens.append(tokens)
                else:
                    for ref_index, ref_latent in enumerate((ip_adapter_latents[batch_index] if ip_adapter_latents is not None else []) or []):
                        if ref_latent.ndim == 4:
                            ref_latent = ref_latent.unsqueeze(2)
                        ref_latent = ref_latent.to(device=x_i.device, dtype=x_i.dtype)
                        tokens, ids, _ = self._prepare_flat_tokens(
                            ref_latent,
                            t_offset=reference_t_offset_scale * (ref_index + 1),
                            padding_mask=None,
                        )
                        visual_latent_tokens.append(tokens)
                        visual_latent_ids.append(ids)

                extra_tokens = ref_tokens + visual_latent_tokens
                extra_ids = ref_ids + visual_latent_ids
                if extra_tokens:
                    x_tokens = torch.cat([target_tokens] + extra_tokens, dim=1)
                    ids = torch.cat([target_ids] + extra_ids, dim=0)
                else:
                    x_tokens = target_tokens
                    ids = target_ids

                rope_emb = self.pos_embedder.generate_embeddings_from_ids(ids, fps=None)
                timestep_i = timesteps_B_T[batch_index : batch_index + 1, :1]
                t_embedding, adaln_lora = self.t_embedder(timestep_i)
                t_embedding = self.t_embedding_norm(t_embedding)
                context_i = crossattn_emb[batch_index : batch_index + 1]
                if visual_context_tokens:
                    context_i = torch.cat([context_i] + visual_context_tokens, dim=1)
                use_fp32 = x_tokens.dtype == torch.float16

                for block_idx, block in enumerate(self.blocks):
                    if self.blocks_to_swap:
                        self.offloader.wait_for_block(block_idx)
                    x_tokens = block.forward_flat(
                        x_tokens,
                        t_embedding,
                        context_i,
                        attn_params,
                        use_fp32,
                        rope_emb,
                        adaln_lora,
                        ip_adapter_tokens=None,
                        ip_adapter_query_len=None,
                    )
                    if self.blocks_to_swap:
                        self.offloader.submit_move_blocks(self.blocks, block_idx)

                target_len = target_tokens.shape[1]
                x_target = x_tokens[:, :target_len]
                T, H, W = target_grid
                x_target = rearrange(x_target, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
                x_out = self.final_layer(x_target, t_embedding, adaln_lora_B_T_3D=adaln_lora, use_fp32=use_fp32)
                outputs.append(self.unpatchify(x_out))

            return torch.cat(outputs, dim=0)

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb,
        }

        attn_params = attention.AttentionParams.create_attention_params(self.attn_mode, self.split_attn)

        # Determine whether to use float32 for block computations based on input dtype (use float32 for better stability when input is float16)
        use_fp32 = x_B_T_H_W_D.dtype == torch.float16

        for block_idx, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(block_idx)

            x_B_T_H_W_D = block(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, attn_params, use_fp32, **block_kwargs)

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks(self.blocks, block_idx)

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D, use_fp32=use_fp32)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        return x_B_C_Tt_Hp_Wp

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        target_input_ids: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
        reference_latents: Optional[list[list[torch.Tensor]]] = None,
        reference_t_offset_scale: int = 10,
        **kwargs,
    ) -> torch.Tensor:
        context = self._preprocess_text_embeds(context, target_input_ids, target_attention_mask, source_attention_mask)
        return self.forward_mini_train_dit(
            x,
            timesteps,
            context,
            fps=fps,
            padding_mask=padding_mask,
            reference_latents=reference_latents,
            reference_t_offset_scale=reference_t_offset_scale,
            **kwargs,
        )

    def _preprocess_text_embeds(
        self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None
    ):
        if target_input_ids is not None:
            context = self.llm_adapter(
                source_hidden_states,
                target_input_ids,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
            )
            context[~target_attention_mask.bool()] = 0  # zero out padding tokens
            return context
        else:
            return source_hidden_states


# LLM Adapter: Bridges Qwen3 embeddings to T5-compatible space
class LLMAdapterRMSNorm(nn.Module):
    """RMSNorm specifically for the LLM Adapter (T5-style, no mean subtraction)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


def _adapter_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _adapter_apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (_adapter_rotate_half(x) * sin)
    return x_embed


class AdapterRotaryEmbedding(nn.Module):
    """Rotary embedding for LLM Adapter."""

    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class AdapterImageRotaryEmbedding(nn.Module):
    """2D rotary embedding for visual token grids."""

    def __init__(self, head_dim: int, rope_theta: float = 256.0):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        half_dim = head_dim // 2
        if half_dim % 2 != 0:
            raise ValueError(f"AdapterImageRotaryEmbedding requires head_dim/2 to be even, got head_dim={head_dim}.")
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int], num_images: int = 1):
        height, width = grid_hw
        device = x.device
        rows = torch.arange(height, device=device)
        cols = torch.arange(width, device=device)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        pos_h = yy.reshape(-1).repeat(num_images)
        pos_w = xx.reshape(-1).repeat(num_images)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs_h = torch.outer(pos_h.float(), self.inv_freq.float())
            freqs_w = torch.outer(pos_w.float(), self.inv_freq.float())
            emb_h = torch.cat((freqs_h, freqs_h), dim=-1)
            emb_w = torch.cat((freqs_w, freqs_w), dim=-1)
            emb = torch.cat((emb_h, emb_w), dim=-1)
            cos = emb.cos().unsqueeze(0)
            sin = emb.sin().unsqueeze(0)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LLMAdapterAttention(nn.Module):
    """Attention module for LLM Adapter with QK-norm and separate RoPE for query/key."""

    def __init__(self, query_dim, context_dim, n_heads, head_dim, norm_eps=1e-6):
        super().__init__()

        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = LLMAdapterRMSNorm(self.head_dim, eps=norm_eps)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = LLMAdapterRMSNorm(self.head_dim, eps=norm_eps)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)

        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            assert position_embeddings_context is not None
            cos, sin = position_embeddings
            query_states = _adapter_apply_rotary_pos_emb(query_states, cos, sin)
            cos, sin = position_embeddings_context
            key_states = _adapter_apply_rotary_pos_emb(key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class OmniFeedForward(nn.Module):
    """SwiGLU feed-forward used by the omni visual refiner."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class OmniRefinerBlock(nn.Module):
    """Single-stream visual token refiner for omni visual features.

    The refiner is attention + SwiGLU FFN with RMSNorm before each branch and
    RMSNorm on branch outputs before residual addition.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 8.0 / 3.0, norm_eps: float = 1e-6) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"OmniRefinerBlock dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.attention_norm1 = LLMAdapterRMSNorm(dim, eps=norm_eps)
        self.attention = LLMAdapterAttention(
            query_dim=dim,
            context_dim=dim,
            n_heads=num_heads,
            head_dim=dim // num_heads,
            norm_eps=norm_eps,
        )
        self.attention_norm2 = LLMAdapterRMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = LLMAdapterRMSNorm(dim, eps=norm_eps)
        self.feed_forward = OmniFeedForward(dim=dim, hidden_dim=int(dim * mlp_ratio))
        self.ffn_norm2 = LLMAdapterRMSNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, position_embeddings=None, attention_mask=None) -> torch.Tensor:
        attn_out = self.attention(
            self.attention_norm1(x),
            mask=attention_mask,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings,
        )
        x = x + self.attention_norm2(attn_out)
        x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x

    def init_weights(self) -> None:
        dim = self.attention.query_dim
        hidden_dim = self.feed_forward.w1.out_features
        std = 1.0 / math.sqrt(dim)
        torch.nn.init.trunc_normal_(self.attention.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.attention.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.attention.v_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.feed_forward.w1.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.feed_forward.w3.weight, std=std, a=-3 * std, b=3 * std)
        out_std = 1.0 / math.sqrt(hidden_dim)
        torch.nn.init.trunc_normal_(self.attention.o_proj.weight, std=out_std, a=-3 * out_std, b=3 * out_std)
        torch.nn.init.trunc_normal_(self.feed_forward.w2.weight, std=out_std, a=-3 * out_std, b=3 * out_std)


class LLMAdapterTransformerBlock(nn.Module):
    """Transformer block for LLM Adapter: optional self-attn + cross-attn + MLP."""

    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, self_attn=False, layer_norm=False):
        super().__init__()
        self.has_self_attn = self_attn

        if self.has_self_attn:
            self.norm_self_attn = nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
            self.self_attn = LLMAdapterAttention(
                query_dim=model_dim,
                context_dim=model_dim,
                n_heads=num_heads,
                head_dim=model_dim // num_heads,
            )

        self.norm_cross_attn = nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
        self.cross_attn = LLMAdapterAttention(
            query_dim=model_dim,
            context_dim=source_dim,
            n_heads=num_heads,
            head_dim=model_dim // num_heads,
        )

        self.norm_mlp = nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio)), nn.GELU(), nn.Linear(int(model_dim * mlp_ratio), model_dim)
        )

    def forward(
        self,
        x,
        context,
        target_attention_mask=None,
        source_attention_mask=None,
        position_embeddings=None,
        position_embeddings_context=None,
    ):
        if self.has_self_attn:
            # Self-attention: target_attention_mask is not expected to be all zeros
            normed = self.norm_self_attn(x)
            attn_out = self.self_attn(
                normed,
                mask=target_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings,
            )
            x = x + attn_out

        normed = self.norm_cross_attn(x)
        attn_out = self.cross_attn(
            normed,
            mask=source_attention_mask,
            context=context,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings_context,
        )
        x = x + attn_out

        x = x + self.mlp(self.norm_mlp(x))
        return x

    def init_weights(self):
        torch.nn.init.zeros_(self.mlp[2].weight)


class LLMAdapter(nn.Module):
    """Bridge module: Qwen3 embeddings (source) → T5-compatible space (target).

    Uses T5 token IDs as target input, embeds them, and cross-attends to Qwen3 hidden states.
    """

    def __init__(
        self, source_dim, target_dim, model_dim, num_layers=6, num_heads=16, embed=None, self_attn=False, layer_norm=False
    ):
        super().__init__()
        if embed is not None:
            self.embed = nn.Embedding.from_pretrained(embed.weight)
        else:
            self.embed = nn.Embedding(32128, target_dim)
        if model_dim != target_dim:
            self.in_proj = nn.Linear(target_dim, model_dim)
        else:
            self.in_proj = nn.Identity()
        self.rotary_emb = AdapterRotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList(
            [
                LLMAdapterTransformerBlock(source_dim, model_dim, num_heads=num_heads, self_attn=self_attn, layer_norm=layer_norm)
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = LLMAdapterRMSNorm(target_dim)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)

        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        x = self.in_proj(self.embed(target_input_ids))
        context = source_hidden_states
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(
                x,
                context,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings_context,
            )
        return self.norm(self.out_proj(x))


# Not used currently, but kept for reference

# def get_dit_config(state_dict, key_prefix=""):
#     """Derive DiT configuration from state_dict weight shapes."""
#     dit_config = {}
#     dit_config["max_img_h"] = 512
#     dit_config["max_img_w"] = 512
#     dit_config["max_frames"] = 128
#     concat_padding_mask = True
#     dit_config["in_channels"] = (state_dict["{}x_embedder.proj.1.weight".format(key_prefix)].shape[1] // 4) - int(
#         concat_padding_mask
#     )
#     dit_config["out_channels"] = 16
#     dit_config["patch_spatial"] = 2
#     dit_config["patch_temporal"] = 1
#     dit_config["model_channels"] = state_dict["{}x_embedder.proj.1.weight".format(key_prefix)].shape[0]
#     dit_config["concat_padding_mask"] = concat_padding_mask
#     dit_config["crossattn_emb_channels"] = 1024
#     dit_config["pos_emb_cls"] = "rope3d"
#     dit_config["pos_emb_learnable"] = True
#     dit_config["pos_emb_interpolation"] = "crop"
#     dit_config["min_fps"] = 1
#     dit_config["max_fps"] = 30

#     dit_config["use_adaln_lora"] = True
#     dit_config["adaln_lora_dim"] = 256
#     if dit_config["model_channels"] == 2048:
#         dit_config["num_blocks"] = 28
#         dit_config["num_heads"] = 16
#     elif dit_config["model_channels"] == 5120:
#         dit_config["num_blocks"] = 36
#         dit_config["num_heads"] = 40
#     elif dit_config["model_channels"] == 1280:
#         dit_config["num_blocks"] = 20
#         dit_config["num_heads"] = 20

#     if dit_config["in_channels"] == 16:
#         dit_config["extra_per_block_abs_pos_emb"] = False
#         dit_config["rope_h_extrapolation_ratio"] = 4.0
#         dit_config["rope_w_extrapolation_ratio"] = 4.0
#         dit_config["rope_t_extrapolation_ratio"] = 1.0
#     elif dit_config["in_channels"] == 17:
#         dit_config["extra_per_block_abs_pos_emb"] = False
#         dit_config["rope_h_extrapolation_ratio"] = 3.0
#         dit_config["rope_w_extrapolation_ratio"] = 3.0
#         dit_config["rope_t_extrapolation_ratio"] = 1.0

#     dit_config["extra_h_extrapolation_ratio"] = 1.0
#     dit_config["extra_w_extrapolation_ratio"] = 1.0
#     dit_config["extra_t_extrapolation_ratio"] = 1.0
#     dit_config["rope_enable_fps_modulation"] = False

#     return dit_config
