import argparse
import logging
import os
import sys

import torch

from alice.utils.checkpoint import (
    convert_to_safetensors,
    shard_checkpoint,
    merge_sharded_checkpoint,
    load_checkpoint_auto,
    save_safetensors,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Alice model checkpoints between formats"
    )
    subparsers = parser.add_subparsers(dest='command', help='Conversion command')

    to_st = subparsers.add_parser(
        'to-safetensors',
        help='Convert .pth/.pt to .safetensors')
    to_st.add_argument('--input', type=str, required=True)
    to_st.add_argument('--output', type=str, required=True)

    shard = subparsers.add_parser(
        'shard',
        help='Shard a checkpoint into multiple files')
    shard.add_argument('--input', type=str, required=True)
    shard.add_argument('--output_dir', type=str, required=True)
    shard.add_argument(
        '--max_shard_size', type=float, default=4.0,
        help='Maximum shard size in GB')

    merge = subparsers.add_parser(
        'merge',
        help='Merge sharded checkpoints into a single file')
    merge.add_argument('--index_file', type=str, required=True)
    merge.add_argument('--output', type=str, required=True)

    info = subparsers.add_parser(
        'info',
        help='Print checkpoint information')
    info.add_argument('--input', type=str, required=True)

    return parser.parse_args()


def cmd_to_safetensors(args):
    convert_to_safetensors(
        args.input, args.output,
        metadata={'converted_by': 'alice'})


def cmd_shard(args):
    logging.info(f'Loading checkpoint from {args.input}')
    state_dict = load_checkpoint_auto(args.input)
    shard_checkpoint(
        state_dict,
        args.output_dir,
        max_shard_size_gb=args.max_shard_size)


def cmd_merge(args):
    state_dict = merge_sharded_checkpoint(args.index_file)
    if args.output.endswith('.safetensors'):
        save_safetensors(state_dict, args.output)
    else:
        torch.save(state_dict, args.output)
    logging.info(f'Merged checkpoint saved to {args.output}')


def cmd_info(args):
    logging.info(f'Loading checkpoint from {args.input}')
    state_dict = load_checkpoint_auto(args.input)

    total_params = 0
    total_bytes = 0
    dtype_counts = {}

    for key, tensor in sorted(state_dict.items()):
        params = tensor.nelement()
        bytes_ = params * tensor.element_size()
        total_params += params
        total_bytes += bytes_

        dtype_str = str(tensor.dtype)
        dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + params

    logging.info("=" * 60)
    logging.info(f"Checkpoint: {args.input}")
    logging.info(f"Total keys: {len(state_dict)}")
    logging.info(f"Total params: {total_params / 1e9:.3f}B")
    logging.info(f"Total size: {total_bytes / 1024**3:.2f} GB")
    logging.info("")
    logging.info("Dtype breakdown:")
    for dtype, count in sorted(dtype_counts.items()):
        logging.info(f"  {dtype}: {count / 1e9:.3f}B params ({count / total_params:.1%})")
    logging.info("=" * 60)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])

    args = _parse_args()
    if args.command is None:
        logging.error("Please specify a command: to-safetensors, shard, merge, info")
        sys.exit(1)

    commands = {
        'to-safetensors': cmd_to_safetensors,
        'shard': cmd_shard,
        'merge': cmd_merge,
        'info': cmd_info,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
