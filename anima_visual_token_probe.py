import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from anima_minimal_inference import IPAdapterFeatureExtractor
from library import anima_utils
from library.utils import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe Anima visual adapter tokens by nearest-neighbor lookup in the LLM adapter text embedding table."
    )
    parser.add_argument("--dit", type=str, required=True, help="Anima DiT checkpoint.")
    parser.add_argument("--ip_adapter_weight", type=str, required=True, help="Anima IP/visual adapter safetensors.")
    parser.add_argument("--reference_image", type=str, nargs="+", required=True, help="Reference image path(s).")
    parser.add_argument("--ip_adapter_feature_backend", type=str, default="ccip_tokens", choices=["ccip", "ccip_tokens", "lsnet"])
    parser.add_argument("--ip_adapter_feature_model", type=str, required=True, help="Feature extractor model dir/checkpoint.")
    parser.add_argument("--ip_adapter_feature_dim", type=int, default=None)
    parser.add_argument("--ip_adapter_num_tokens", type=int, default=4)
    parser.add_argument("--t5_tokenizer_path", type=str, default="configs/t5_old")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_visual_tokens", type=int, default=32, help="Print nearest tokens for at most this many visual tokens.")
    parser.add_argument("--include_special_tokens", action="store_true")
    parser.add_argument("--attn_mode", type=str, default="torch")
    parser.add_argument("--linear-adapter", "--ip_adapter_linear_adapter", dest="linear_adapter", action="store_true")
    parser.add_argument("--mlp-adapter", "--ip_adapter_mlp_adapter", dest="mlp_adapter", action="store_true")
    parser.add_argument(
        "--norm-linear-adapter",
        "--norm_linear_adapter",
        "--ip_adapter_norm_linear_adapter",
        dest="norm_linear_adapter",
        action="store_true",
    )
    parser.add_argument("--omni-adapter", "--omni_adapter", "--ip_adapter_omni_adapter", dest="omni_adapter", action="store_true")
    return parser.parse_args()


def dtype_from_arg(name: str):
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    return torch.bfloat16


def decode_token(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    return text.replace("\n", "\\n")


def build_vocab_mask(tokenizer, vocab_size: int, include_special_tokens: bool, device: torch.device):
    mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
    if include_special_tokens:
        return mask
    special_ids = set()
    for token_id in tokenizer.all_special_ids:
        if token_id is not None and 0 <= int(token_id) < vocab_size:
            special_ids.add(int(token_id))
    if getattr(tokenizer, "pad_token_id", None) is not None:
        special_ids.add(int(tokenizer.pad_token_id))
    if special_ids:
        ids = torch.tensor(sorted(special_ids), dtype=torch.long, device=device)
        mask[ids] = False
    return mask


def topk_tokens(query, embedding, vocab_mask, top_k):
    query = F.normalize(query.float(), dim=-1)
    embedding = F.normalize(embedding.float(), dim=-1)
    scores = query @ embedding.T
    scores[:, ~vocab_mask] = -torch.inf
    return torch.topk(scores, k=top_k, dim=-1)


def print_neighbors(title, values, indices, tokenizer):
    print(f"\n{title}")
    for rank, (score, token_id) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        token = decode_token(tokenizer, int(token_id))
        print(f"  {rank:02d}. id={int(token_id):6d} score={float(score): .4f} token={token!r}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = dtype_from_arg(args.dtype)

    if sum(bool(getattr(args, name)) for name in ("linear_adapter", "mlp_adapter", "norm_linear_adapter", "omni_adapter")) > 1:
        raise ValueError("Only one adapter type flag can be enabled.")
    if not args.omni_adapter and not args.linear_adapter and not args.mlp_adapter and not args.norm_linear_adapter:
        logger.warning("No adapter type flag was passed. If the checkpoint was trained with --omni-adapter, pass --omni-adapter.")

    extractor = IPAdapterFeatureExtractor(args.ip_adapter_feature_backend, args.ip_adapter_feature_model, device, dtype)
    feature_dim = args.ip_adapter_feature_dim or extractor.feature_dim
    if feature_dim is None:
        raise ValueError("Could not infer feature dim. Pass --ip_adapter_feature_dim.")

    logger.info("Loading Anima model and visual adapter.")
    model = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=True,
        loading_device=device,
        dit_weight_dtype=dtype,
        enable_ip_adapter=True,
        ip_adapter_feature_dim=feature_dim,
        ip_adapter_num_tokens=args.ip_adapter_num_tokens,
        ip_adapter_linear_adapter=args.linear_adapter,
        ip_adapter_mlp_adapter=args.mlp_adapter,
        ip_adapter_norm_linear_adapter=args.norm_linear_adapter,
        ip_adapter_omni_adapter=args.omni_adapter,
    )
    info = model.load_state_dict(load_file(args.ip_adapter_weight), strict=False)
    unexpected = [key for key in info.unexpected_keys if "visual_condition_adapter" in key or "ip_adapter" in key]
    if unexpected:
        logger.warning(f"Unexpected adapter keys: {unexpected[:8]}")
    model.eval()

    if not hasattr(model, "llm_adapter") or model.llm_adapter is None:
        raise RuntimeError("This Anima model has no llm_adapter; cannot probe target text embedding space.")
    text_embedding = model.llm_adapter.embed.weight.detach().to(device)
    target_dim = text_embedding.shape[-1]

    features = extractor.extract(args.reference_image).to(device=device, dtype=dtype)
    with torch.no_grad():
        visual_tokens = model.project_ip_adapter_features(features.unsqueeze(0)).squeeze(0)
    if visual_tokens.shape[-1] != target_dim:
        raise RuntimeError(
            f"Visual token dim {visual_tokens.shape[-1]} does not match LLM adapter embedding dim {target_dim}."
        )

    tokenizer = anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)
    vocab_size = min(text_embedding.shape[0], len(tokenizer))
    text_embedding = text_embedding[:vocab_size]
    vocab_mask = build_vocab_mask(tokenizer, vocab_size, args.include_special_tokens, device)

    pooled = visual_tokens.mean(dim=0, keepdim=True)
    pooled_values, pooled_indices = topk_tokens(pooled, text_embedding, vocab_mask, args.top_k)
    print(f"visual_tokens shape: {tuple(visual_tokens.shape)}")
    print_neighbors("pooled visual token nearest text tokens", pooled_values[0], pooled_indices[0], tokenizer)

    per_token_count = min(args.max_visual_tokens, visual_tokens.shape[0])
    values, indices = topk_tokens(visual_tokens[:per_token_count], text_embedding, vocab_mask, args.top_k)
    for i in range(per_token_count):
        print_neighbors(f"visual token #{i:03d} nearest text tokens", values[i], indices[i], tokenizer)


if __name__ == "__main__":
    main()
