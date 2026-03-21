import logging
import math
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..models.vae import AliceVAE


__all__ = ['StreamingDecoder', 'ChunkedLatentIterator']
