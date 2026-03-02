import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .t2v_14b import t2v_14b
from .i2v_14b import i2v_14b

ALICE_CONFIGS = {
    't2v-14b': t2v_14b,
    'i2v-14b': i2v_14b,
}

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
}

MAX_AREA_CONFIGS = {
    '720*1280': 720 * 1280,
    '1280*720': 1280 * 720,
    '480*832': 480 * 832,
    '832*480': 832 * 480,
}

SUPPORTED_SIZES = {
    't2v-14b': ('720*1280', '1280*720', '480*832', '832*480'),
    'i2v-14b': ('720*1280', '1280*720', '480*832', '832*480'),
}
