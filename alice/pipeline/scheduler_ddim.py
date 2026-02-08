"""DDIM scheduler for flow-matching diffusion models.

Experimental implementation - testing if DDIM provides benefits over DPM++.
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import (
    SchedulerMixin,
    SchedulerOutput,
)


class FlowDDIMScheduler(SchedulerMixin, ConfigMixin):
    """DDIM scheduler for flow-matching diffusion models."""

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        prediction_type: str = "flow_prediction",
        shift: Optional[float] = 1.0,
        clip_sample: bool = False,
        set_alpha_to_one: bool = True,
    ):
        self.num_inference_steps = None
        alphas = np.linspace(1, 1 / num_train_timesteps,
                             num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

    def set_timesteps(self, num_inference_steps: int, device=None):
        """Set discrete timesteps for diffusion chain."""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.config.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self._step_index = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator=None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """Predict sample at previous timestep using DDIM."""
        # Simple DDIM step
        sigma_t = self.sigmas[self._step_index]
        prev_sample = sample - sigma_t * model_output

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)
