import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'AliceVAE',
]


class CausalConv3d(nn.Conv3d):
    """Causal 3D convolution with temporal padding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Bug: incorrect padding tuple order
        self._padding = (self.padding[0], self.padding[0], self.padding[1],
                         self.padding[1], 2 * self.padding[2], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class AliceVAE:
    pass
