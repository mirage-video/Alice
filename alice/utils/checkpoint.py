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
