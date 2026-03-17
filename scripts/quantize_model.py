import argparse
import logging
import os
import sys

import torch

from alice.quantization import quantize_model, FP8Linear
from alice.quantization.calibration import CalibrationContext
from alice.quantization.utils import get_fp8_info, estimate_quantization_error
from alice.models.transformer import AliceTransformer
from alice.utils.checkpoint import save_safetensors, load_checkpoint_auto


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize Alice model weights to FP8 for efficient inference"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the model checkpoint.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the quantized model.")
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder within input_dir for the model (e.g. low_noise_model).")
    parser.add_argument(
        "--skip_modules",
        type=str,
        nargs='*',
        default=['head.head', 'patch_embedding', 'text_embedding.0', 'text_embedding.2'],
        help="Module names to skip during quantization.")
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Run quantization error analysis on a sample of weights.")
    return parser.parse_args()


def main():
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])

    fp8_info = get_fp8_info()
    logging.info(f"FP8 E4M3 range: [{fp8_info['e4m3fn']['min']}, {fp8_info['e4m3fn']['max']}]")

    logging.info(f"Loading model from {args.input_dir}")
    if args.subfolder:
        model_path = os.path.join(args.input_dir, args.subfolder)
    else:
        model_path = args.input_dir

    model = AliceTransformer.from_pretrained(model_path)
    model.eval()

    orig_params = sum(p.nelement() for p in model.parameters())
    orig_size = sum(
        p.nelement() * p.element_size() for p in model.parameters())
    logging.info(
        f"Original model: {orig_params / 1e9:.2f}B params, "
        f"{orig_size / 1024**3:.2f} GB")

    if args.verify:
        logging.info("Running quantization error analysis...")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name not in args.skip_modules:
                w = module.weight.data.float()
                from alice.quantization.utils import compute_scale
                scale = compute_scale(w, torch.finfo(torch.float8_e4m3fn).max)
                error = estimate_quantization_error(w, scale)
                logging.info(
                    f"  {name}: SNR={error['snr_db']:.1f}dB, "
                    f"MAE={error['mae']:.6f}, MSE={error['mse']:.8f}")

    logging.info("Quantizing model to FP8...")
    model = quantize_model(
        model,
        skip_modules=args.skip_modules)

    quant_size = sum(
        p.nelement() * p.element_size() for p in model.parameters())
    buf_size = sum(
        b.nelement() * b.element_size() for b in model.buffers())
    total_size = quant_size + buf_size
    logging.info(
        f"Quantized model: {total_size / 1024**3:.2f} GB "
        f"({1 - total_size / orig_size:.1%} reduction)")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'model_fp8.safetensors')

    state_dict = model.state_dict()
    try:
        save_safetensors(
            state_dict, output_path,
            metadata={
                'format': 'fp8_e4m3fn',
                'source': args.input_dir,
                'skipped_modules': ','.join(args.skip_modules),
            })
    except ImportError:
        torch.save(state_dict, output_path.replace('.safetensors', '.pth'))
        logging.info("safetensors not available, saved as .pth")

    logging.info(f"Quantized model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
