import argparse
import logging

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from anima_minimal_inference import IPAdapterFeatureExtractor
from library import anima_utils
from library.utils import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Experimental decoder for Anima visual adapter tokens: optimize a Qwen-side hidden sequence whose "
            "LLM adapter output reconstructs visual tokens, then use Qwen with optimized soft input embeddings to generate text."
        )
    )
    parser.add_argument("--dit", type=str, required=True)
    parser.add_argument("--qwen3", "--text_encoder", dest="qwen3", type=str, required=True)
    parser.add_argument("--ip_adapter_weight", type=str, required=True)
    parser.add_argument("--reference_image", type=str, nargs="+", required=True)
    parser.add_argument("--ip_adapter_feature_backend", type=str, default="ccip_tokens", choices=["ccip", "ccip_tokens", "lsnet"])
    parser.add_argument("--ip_adapter_feature_model", type=str, required=True)
    parser.add_argument("--ip_adapter_feature_dim", type=int, default=None)
    parser.add_argument("--ip_adapter_num_tokens", type=int, default=4)
    parser.add_argument("--t5_tokenizer_path", type=str, default="configs/t5_old")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--attn_mode", type=str, default="torch")
    parser.add_argument("--omni-adapter", "--omni_adapter", "--ip_adapter_omni_adapter", dest="omni_adapter", action="store_true")
    parser.add_argument("--linear-adapter", "--ip_adapter_linear_adapter", dest="linear_adapter", action="store_true")
    parser.add_argument("--mlp-adapter", "--ip_adapter_mlp_adapter", dest="mlp_adapter", action="store_true")
    parser.add_argument("--norm-linear-adapter", "--norm_linear_adapter", dest="norm_linear_adapter", action="store_true")
    parser.add_argument("--source_len", type=int, default=64, help="Length of optimized Qwen-side source hidden sequence.")
    parser.add_argument("--max_visual_tokens", type=int, default=64, help="Use at most this many visual tokens for inversion.")
    parser.add_argument("--invert_steps", type=int, default=800)
    parser.add_argument("--invert_lr", type=float, default=0.03)
    parser.add_argument("--embed_steps", type=int, default=500)
    parser.add_argument("--embed_lr", type=float, default=0.02)
    parser.add_argument("--generate_max_new_tokens", type=int, default=96)
    parser.add_argument("--prompt", type=str, default="Describe this image in concise danbooru-style tags:")
    parser.add_argument("--skip_soft_generate", action="store_true")
    return parser.parse_args()


def dtype_from_arg(name: str):
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    return torch.bfloat16


def nearest_t5_ids(visual_tokens, embed_weight, top_fallback_id=3):
    visual = F.normalize(visual_tokens.float(), dim=-1)
    embed = F.normalize(embed_weight.float(), dim=-1)
    scores = visual @ embed.T
    ids = scores.argmax(dim=-1)
    # Avoid pad/eos dominating very early failed adapters.
    ids = torch.where(ids <= 1, torch.full_like(ids, top_fallback_id), ids)
    return ids


def main():
    args = parse_args()
    if sum(bool(getattr(args, name)) for name in ("linear_adapter", "mlp_adapter", "norm_linear_adapter", "omni_adapter")) > 1:
        raise ValueError("Only one adapter type flag can be enabled.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = dtype_from_arg(args.dtype)

    extractor = IPAdapterFeatureExtractor(args.ip_adapter_feature_backend, args.ip_adapter_feature_model, device, dtype)
    feature_dim = args.ip_adapter_feature_dim or extractor.feature_dim
    if feature_dim is None:
        raise ValueError("Could not infer feature dim. Pass --ip_adapter_feature_dim.")

    logger.info("Loading Anima model and adapter.")
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
    model.load_state_dict(load_file(args.ip_adapter_weight), strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    logger.info("Extracting visual adapter tokens.")
    features = extractor.extract(args.reference_image).to(device=device, dtype=dtype)
    with torch.no_grad():
        visual_tokens = model.project_ip_adapter_features(features.unsqueeze(0)).squeeze(0).float()
    if args.max_visual_tokens > 0:
        visual_tokens = visual_tokens[: args.max_visual_tokens]

    llm_adapter = model.llm_adapter
    target_embed = llm_adapter.embed.weight.detach().to(device)
    target_ids = nearest_t5_ids(visual_tokens.to(device), target_embed).unsqueeze(0).to(device)
    target_mask = torch.ones_like(target_ids, dtype=torch.bool, device=device)
    target = visual_tokens.unsqueeze(0).to(device)

    source_dim = llm_adapter.blocks[0].cross_attn.context_dim
    source_len = args.source_len
    source_hidden = torch.randn((1, source_len, source_dim), device=device, dtype=torch.float32) * 0.02
    source_hidden.requires_grad_(True)
    source_mask = torch.ones((1, source_len), dtype=torch.bool, device=device)
    optimizer = torch.optim.AdamW([source_hidden], lr=args.invert_lr)

    logger.info("Optimizing inverse LLM-adapter source hidden states.")
    for step in range(args.invert_steps):
        optimizer.zero_grad(set_to_none=True)
        pred = llm_adapter(
            source_hidden_states=source_hidden.to(dtype),
            target_input_ids=target_ids,
            target_attention_mask=target_mask,
            source_attention_mask=source_mask,
        ).float()
        recon = F.mse_loss(pred, target)
        smooth = (source_hidden[:, 1:] - source_hidden[:, :-1]).pow(2).mean()
        norm_reg = source_hidden.pow(2).mean()
        loss = recon + 0.01 * smooth + 0.001 * norm_reg
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.invert_steps - 1:
            logger.info(f"invert step={step:04d} loss={loss.item():.6f} recon={recon.item():.6f}")

    tokenizer = AutoTokenizer.from_pretrained(args.qwen3, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qwen = AutoModelForCausalLM.from_pretrained(args.qwen3, torch_dtype=dtype, local_files_only=True).to(device).eval()
    for p in qwen.parameters():
        p.requires_grad_(False)

    # A rough textual fallback: nearest input embedding tokens to the optimized final hidden states.
    input_embed = qwen.get_input_embeddings().weight.detach().to(device)
    nearest = (F.normalize(source_hidden.detach()[0].float(), dim=-1) @ F.normalize(input_embed.float(), dim=-1).T).argmax(dim=-1)
    nearest_text = tokenizer.decode(nearest.tolist(), skip_special_tokens=True)
    print("\nnearest Qwen-token approximation:")
    print(nearest_text.strip())

    if args.skip_soft_generate:
        return

    logger.info("Optimizing Qwen input embeddings to reproduce the inverted source hidden states.")
    prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    prompt_embeds = qwen.get_input_embeddings()(prompt_ids).detach()
    soft_embeds = torch.randn((1, source_len, input_embed.shape[-1]), device=device, dtype=torch.float32) * 0.02
    soft_embeds.requires_grad_(True)
    embed_optimizer = torch.optim.AdamW([soft_embeds], lr=args.embed_lr)
    target_source = source_hidden.detach().to(device).float()

    for step in range(args.embed_steps):
        embed_optimizer.zero_grad(set_to_none=True)
        outputs = qwen.model(inputs_embeds=soft_embeds.to(dtype), attention_mask=source_mask, output_hidden_states=False)
        pred_hidden = outputs.last_hidden_state.float()
        loss = F.mse_loss(pred_hidden, target_source) + 0.001 * soft_embeds.pow(2).mean()
        loss.backward()
        embed_optimizer.step()
        if step % 100 == 0 or step == args.embed_steps - 1:
            logger.info(f"embed step={step:04d} loss={loss.item():.6f}")

    prefix_embeds = torch.cat([prompt_embeds, soft_embeds.detach().to(dtype)], dim=1)
    attention_mask = torch.ones(prefix_embeds.shape[:2], dtype=torch.long, device=device)
    with torch.no_grad():
        out = qwen.generate(
            inputs_embeds=prefix_embeds,
            attention_mask=attention_mask,
            max_new_tokens=args.generate_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print("\nQwen soft-prefix decode:")
    print(tokenizer.decode(out[0], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
