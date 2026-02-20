import json
import logging
import math
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional, Union

import dashscope
import torch
from PIL import Image

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_VER = 2
except ModuleNotFoundError:
    flash_attn_varlen_func = None  # in compatible with CPU machines
    FLASH_VER = None


@dataclass
class PromptOutput(object):
    status: bool
    prompt: str
    seed: int
    system_prompt: str
    message: str

    def add_custom_field(self, key: str, value) -> None:
        self.__setattr__(key, value)
