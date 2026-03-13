import logging
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .utils import compute_scale


FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
