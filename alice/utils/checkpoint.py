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
