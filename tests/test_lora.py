import pytest
import torch
import torch.nn as nn

from alice.adapters.lora import LoRALinear, LoRAAdapter, merge_lora_weights


class TestLoRALinear:

    def test_init(self):
        base = nn.Linear(128, 256)
        lora = LoRALinear(base, rank=8, alpha=16.0)
        assert lora.rank == 8
        assert lora.alpha == 16.0
        assert lora.scaling == 16.0 / 8

    def test_forward_shape(self):
        base = nn.Linear(64, 32)
        lora = LoRALinear(base, rank=4)
        x = torch.randn(2, 64)
        out = lora(x)
        assert out.shape == (2, 32)

    def test_forward_adds_residual(self):
        base = nn.Linear(64, 32)
        lora = LoRALinear(base, rank=4)
        nn.init.ones_(lora.lora_b.weight)
        x = torch.randn(2, 64)
        base_out = base(x)
        lora_out = lora(x)
        assert not torch.allclose(base_out, lora_out)

    def test_merge(self):
        base = nn.Linear(64, 32, bias=True)
        lora = LoRALinear(base, rank=4)
        merged = lora.merge()
        assert isinstance(merged, nn.Linear)
        assert merged.weight.shape == (32, 64)
        assert merged.bias is not None
        assert merged.bias.shape == (32,)


class TestLoRAAdapter:

    def test_apply(self):
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )
        adapter = LoRAAdapter(rank=8, target_modules=['0', '2'])
        adapter.apply(model)
        assert isinstance(model[0], LoRALinear)
        assert isinstance(model[2], LoRALinear)

    def test_merge_lora_weights(self):
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )
        adapter = LoRAAdapter(rank=4, target_modules=['0', '2'])
        adapter.apply(model)
        assert isinstance(model[0], LoRALinear)
        merge_lora_weights(model)
        assert isinstance(model[0], nn.Linear)
        assert isinstance(model[2], nn.Linear)
