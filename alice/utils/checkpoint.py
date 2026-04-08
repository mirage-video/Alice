import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from safetensors.torch import load_file, save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


__all__ = [
    'save_safetensors',
    'load_safetensors',
    'convert_to_safetensors',
    'shard_checkpoint',
    'merge_sharded_checkpoint',
    'load_checkpoint_auto',
]


def save_safetensors(
    state_dict: Dict[str, torch.Tensor],
    path: str,
    metadata: Optional[Dict[str, str]] = None,
):
    if not SAFETENSORS_AVAILABLE:
        raise ImportError('safetensors is required: pip install safetensors')

    tensors = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            tensors[k] = v.contiguous().cpu()

    save_file(tensors, path, metadata=metadata)
    logging.info(f'Saved {len(tensors)} tensors to {path}')


def load_safetensors(
    path: str,
    device: str = 'cpu',
) -> Dict[str, torch.Tensor]:
    if not SAFETENSORS_AVAILABLE:
        raise ImportError('safetensors is required: pip install safetensors')

    state_dict = load_file(path, device=device)
    logging.info(f'Loaded {len(state_dict)} tensors from {path}')
    return state_dict


def convert_to_safetensors(
    input_path: str,
    output_path: str,
    metadata: Optional[Dict[str, str]] = None,
):
    logging.info(f'Converting {input_path} -> {output_path}')
    state_dict = torch.load(input_path, map_location='cpu')

    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    save_safetensors(state_dict, output_path, metadata=metadata)


def shard_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    output_dir: str,
    max_shard_size_gb: float = 4.0,
    prefix: str = 'model',
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    max_bytes = int(max_shard_size_gb * 1024 ** 3)

    shards = []
    current_shard = {}
    current_size = 0

    sorted_keys = sorted(
        state_dict.keys(),
        key=lambda k: state_dict[k].nelement() * state_dict[k].element_size(),
        reverse=True)

    for key in sorted_keys:
        tensor = state_dict[key]
        tensor_size = tensor.nelement() * tensor.element_size()

        if current_size + tensor_size > max_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    index = {'metadata': {'total_size': 0}, 'weight_map': {}}

    for i, shard in enumerate(shards):
        shard_name = f'{prefix}-{i + 1:05d}-of-{len(shards):05d}.safetensors'
        shard_path = os.path.join(output_dir, shard_name)

        if SAFETENSORS_AVAILABLE:
            save_safetensors(shard, shard_path)
        else:
            torch.save(shard, shard_path)

        for key in shard:
            index['weight_map'][key] = shard_name
            index['metadata']['total_size'] += (
                shard[key].nelement() * shard[key].element_size())

    index_path = os.path.join(output_dir, f'{prefix}.safetensors.index.json')
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    logging.info(
        f'Sharded into {len(shards)} files '
        f'({index["metadata"]["total_size"] / 1024**3:.1f} GB total)')
    return index


def merge_sharded_checkpoint(
    index_path: str,
    device: str = 'cpu',
) -> Dict[str, torch.Tensor]:
    with open(index_path, 'r') as f:
        index = json.load(f)

    base_dir = os.path.dirname(index_path)
    shard_files = set(index['weight_map'].values())

    merged = {}
    for shard_file in sorted(shard_files):
        shard_path = os.path.join(base_dir, shard_file)
        if shard_path.endswith('.safetensors') and SAFETENSORS_AVAILABLE:
            shard_dict = load_safetensors(shard_path, device=device)
        else:
            shard_dict = torch.load(shard_path, map_location=device)
        merged.update(shard_dict)

    logging.info(f'Merged {len(shard_files)} shards into {len(merged)} tensors')
    return merged
