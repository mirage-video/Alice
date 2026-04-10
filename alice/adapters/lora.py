import logging
import re
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['LoRAAdapter', 'inject_lora', 'extract_lora', 'merge_lora_weights']


class LoRALinear(nn.Module):
    pass
