import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_rms_norm_shape():
    from alice.models.transformer import RMSNorm

    dim = 32
    batch, seq = 2, 10
    norm = RMSNorm(dim)

    x = torch.randn(batch, seq, dim)
    out = norm(x)

    assert out.shape == (batch, seq, dim)


def test_layer_norm_shape():
    from alice.models.transformer import LayerNorm

    dim = 32
    batch, seq = 2, 10
    norm = LayerNorm(dim)

    x = torch.randn(batch, seq, dim)
    out = norm(x)

    assert out.shape == (batch, seq, dim)


def test_sinusoidal_embedding_shape():
    from alice.models.transformer import sinusoidal_embedding_1d

    dim = 64
    seq_len = 16

    position = torch.arange(seq_len)
    emb = sinusoidal_embedding_1d(dim, position)

    assert emb.shape == (seq_len, dim)


def test_rope_params_shape():
    from alice.models.transformer import rope_params

    max_seq = 128
    dim = 32

    freqs = rope_params(max_seq, dim)

    assert freqs.shape == (max_seq, dim // 2)
    assert freqs.dtype == torch.complex128


def test_self_attention_init():
    from alice.models.transformer import SelfAttention

    dim = 64
    num_heads = 4
    attn = SelfAttention(dim, num_heads)

    assert attn.dim == dim
    assert attn.num_heads == num_heads
    assert attn.head_dim == dim // num_heads
    assert attn.q.in_features == dim
    assert attn.k.in_features == dim
    assert attn.v.in_features == dim
    assert attn.o.out_features == dim


def test_cross_attention_init():
    from alice.models.transformer import CrossAttention

    dim = 64
    num_heads = 4
    attn = CrossAttention(dim, num_heads)

    assert attn.dim == dim
    assert attn.num_heads == num_heads


def test_attention_block_init():
    from alice.models.transformer import AttentionBlock

    dim = 64
    ffn_dim = 256
    num_heads = 4
    block = AttentionBlock(dim, ffn_dim, num_heads)

    assert block.dim == dim
    assert block.ffn_dim == ffn_dim
    assert block.num_heads == num_heads
    assert block.modulation.shape == (1, 6, dim)


def test_head_init():
    from alice.models.transformer import Head

    dim = 64
    out_dim = 16
    patch_size = (1, 2, 2)
    head = Head(dim, out_dim, patch_size)

    assert head.dim == dim
    assert head.out_dim == out_dim
    assert head.patch_size == patch_size
    assert head.modulation.shape == (1, 2, dim)


def test_transformer_init_tiny():
    from alice.models.transformer import AliceTransformer

    model = AliceTransformer(
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=16,
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=32,
        text_dim=64,
        out_dim=4,
        num_heads=2,
        num_layers=1,
    )

    assert model.dim == 32
    assert model.num_heads == 2
    assert model.num_layers == 1
    assert len(model.blocks) == 1


def test_transformer_config_registration():
    from alice.models.transformer import AliceTransformer

    model = AliceTransformer(
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=16,
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=32,
        text_dim=64,
        out_dim=4,
        num_heads=2,
        num_layers=1,
    )

    config = model.config
    assert config['dim'] == 32
    assert config['num_heads'] == 2
    assert config['num_layers'] == 1


def test_vae_causal_conv3d_shape():
    from alice.models.vae import CausalConv3d

    in_ch, out_ch = 4, 8
    conv = CausalConv3d(in_ch, out_ch, kernel_size=3, padding=1)

    x = torch.randn(1, in_ch, 4, 8, 8)
    out = conv(x)

    assert out.shape[0] == 1
    assert out.shape[1] == out_ch


def test_vae_rms_norm_shape():
    from alice.models.vae import RMS_norm

    dim = 16
    norm = RMS_norm(dim, channel_first=True, images=False)

    x = torch.randn(2, dim, 4, 8, 8)
    out = norm(x)

    assert out.shape == x.shape


def test_scheduler_dpm_init():
    from alice.pipeline.scheduler_dpm import FlowDPMSolverMultistepScheduler

    scheduler = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        shift=1.0,
    )

    assert scheduler.config.num_train_timesteps == 1000


def test_scheduler_unipc_init():
    from alice.pipeline.scheduler_unipc import FlowUniPCMultistepScheduler

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1.0,
    )

    assert scheduler.config.num_train_timesteps == 1000


def test_config_t2v_14b_shapes():
    from alice.configs import ALICE_CONFIGS

    cfg = ALICE_CONFIGS['t2v-14b']

    assert cfg.dim % cfg.num_heads == 0
    head_dim = cfg.dim // cfg.num_heads
    assert head_dim % 2 == 0

    assert cfg.ffn_dim > cfg.dim
    assert cfg.freq_dim > 0
    assert cfg.num_layers > 0
