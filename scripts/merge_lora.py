import argparse
import logging
import os
import sys

import torch

from alice.adapters.lora import inject_lora, merge_lora_weights
from alice.models.transformer import AliceTransformer
from alice.utils.checkpoint import save_safetensors, load_checkpoint_auto


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter weights into base model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to base model directory or checkpoint.")
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA adapter checkpoint.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the merged model.")
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder for the base model (e.g. low_noise_model).")
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank used during training.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=16.0,
        help="LoRA alpha scaling factor.")
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs='*',
        default=None,
        help="Target module names for LoRA injection.")
    return parser.parse_args()


def main():
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])

    model_path = args.base_model
    if args.subfolder:
        model_path = os.path.join(model_path, args.subfolder)

    logging.info(f"Loading base model from {model_path}")
    model = AliceTransformer.from_pretrained(model_path)
    model.eval()

    base_params = sum(p.nelement() for p in model.parameters())
    logging.info(f"Base model: {base_params / 1e9:.2f}B params")

    logging.info(f"Injecting LoRA from {args.lora_path}")
    model = inject_lora(
        model,
        checkpoint_path=args.lora_path,
        rank=args.rank,
        alpha=args.alpha,
        target_modules=args.target_modules)

    logging.info("Merging LoRA weights into base model...")
    model = merge_lora_weights(model)

    merged_params = sum(p.nelement() for p in model.parameters())
    logging.info(f"Merged model: {merged_params / 1e9:.2f}B params")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'model_merged.safetensors')

    state_dict = model.state_dict()
    try:
        save_safetensors(
            state_dict, output_path,
            metadata={
                'base_model': args.base_model,
                'lora_rank': str(args.rank),
                'lora_alpha': str(args.alpha),
            })
    except ImportError:
        output_path = output_path.replace('.safetensors', '.pth')
        torch.save(state_dict, output_path)
        logging.info("safetensors not available, saved as .pth")

    logging.info(f"Merged model saved to {output_path}")


if __name__ == "__main__":
    main()
