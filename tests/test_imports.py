import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import_alice_package():
    import alice
    assert hasattr(alice, 'AliceTextToVideo')


def test_import_configs():
    from alice.configs import ALICE_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
    assert 't2v-14b' in ALICE_CONFIGS
    assert '1280*720' in SIZE_CONFIGS


def test_import_generator():
    from alice.pipeline import AliceTextToVideo
    assert AliceTextToVideo is not None


def test_import_transformer():
    from alice.models.transformer import AliceTransformer
    assert AliceTransformer is not None


def test_import_attention():
    from alice.models.attention import flash_attention
    assert flash_attention is not None


def test_import_text_encoder():
    from alice.models.text_encoder import T5EncoderModel
    assert T5EncoderModel is not None


def test_import_vae():
    from alice.models.vae import AliceVAE
    assert AliceVAE is not None


def test_import_schedulers():
    from alice.pipeline.scheduler_dpm import FlowDPMSolverMultistepScheduler
    from alice.pipeline.scheduler_unipc import FlowUniPCMultistepScheduler
    assert FlowDPMSolverMultistepScheduler is not None
    assert FlowUniPCMultistepScheduler is not None


def test_import_distributed():
    from alice.distributed.fsdp import shard_model
    from alice.distributed.util import get_world_size
    assert shard_model is not None
    assert get_world_size is not None
