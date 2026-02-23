import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import alice
from alice.configs import ALICE_CONFIGS, MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES


EXAMPLE_PROMPT = {
    "t2v-14b": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}


if __name__ == "__main__":
    pass
