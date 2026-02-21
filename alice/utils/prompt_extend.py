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

from .system_prompt import *

DEFAULT_SYS_PROMPTS = {
    "t2v-A14B": {
        "zh": T2V_A14B_ZH_SYS_PROMPT,
        "en": T2V_A14B_EN_SYS_PROMPT,
    },
    "i2v-A14B": {
        "zh": I2V_A14B_ZH_SYS_PROMPT,
        "en": I2V_A14B_EN_SYS_PROMPT,
        "empty": {
            "zh": I2V_A14B_EMPTY_ZH_SYS_PROMPT,
            "en": I2V_A14B_EMPTY_EN_SYS_PROMPT,
        }
    },
    "ti2v-5B": {
        "t2v": {
            "zh": T2V_A14B_ZH_SYS_PROMPT,
            "en": T2V_A14B_EN_SYS_PROMPT,
        },
        "i2v": {
            "zh": I2V_A14B_ZH_SYS_PROMPT,
            "en": I2V_A14B_EN_SYS_PROMPT,
        }
    },
}


@dataclass
class PromptOutput(object):
    status: bool
    prompt: str
    seed: int
    system_prompt: str
    message: str

    def add_custom_field(self, key: str, value) -> None:
        self.__setattr__(key, value)


class PromptExpander:

    def __init__(self, model_name, task, is_vl=False, device=0, **kwargs):
        self.model_name = model_name
        self.task = task
        self.is_vl = is_vl
        self.device = device

    def extend_with_img(self,
                        prompt,
                        system_prompt,
                        image=None,
                        seed=-1,
                        *args,
                        **kwargs):
        pass

    def extend(self, prompt, system_prompt, seed=-1, *args, **kwargs):
        pass

    def decide_system_prompt(self, tar_lang="zh", prompt=None):
        assert self.task is not None
        if "ti2v" in self.task:
            if self.is_vl:
                return DEFAULT_SYS_PROMPTS[self.task]["i2v"][tar_lang]
            else:
                return DEFAULT_SYS_PROMPTS[self.task]["t2v"][tar_lang]
        if "i2v" in self.task and len(prompt) == 0:
            return DEFAULT_SYS_PROMPTS[self.task]["empty"][tar_lang]
        return DEFAULT_SYS_PROMPTS[self.task][tar_lang]

    def __call__(self,
                 prompt,
                 system_prompt=None,
                 tar_lang="zh",
                 image=None,
                 seed=-1,
                 *args,
                 **kwargs):
        if system_prompt is None:
            system_prompt = self.decide_system_prompt(
                tar_lang=tar_lang, prompt=prompt)
        if seed < 0:
            seed = random.randint(0, sys.maxsize)
        if image is not None and self.is_vl:
            return self.extend_with_img(
                prompt, system_prompt, image=image, seed=seed, *args, **kwargs)
        elif not self.is_vl:
            return self.extend(prompt, system_prompt, seed, *args, **kwargs)
        else:
            raise NotImplementedError


class DashScopePromptExpander(PromptExpander):

    def __init__(self,
                 api_key=None,
                 model_name=None,
                 task=None,
                 max_image_size=512 * 512,
                 retry_times=4,
                 is_vl=False,
                 **kwargs):
        """Initialize DashScope prompt expander with API credentials."""
        if model_name is None:
            model_name = 'qwen-plus' if not is_vl else 'qwen-vl-max'
        super().__init__(model_name, task, is_vl, **kwargs)
        if api_key is not None:
            dashscope.api_key = api_key
        elif 'DASH_API_KEY' in os.environ and os.environ[
                'DASH_API_KEY'] is not None:
            dashscope.api_key = os.environ['DASH_API_KEY']
        else:
            raise ValueError("DASH_API_KEY is not set")
        if 'DASH_API_URL' in os.environ and os.environ[
                'DASH_API_URL'] is not None:
            dashscope.base_http_api_url = os.environ['DASH_API_URL']
        else:
            dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
        self.api_key = api_key

        self.max_image_size = max_image_size
        self.model = model_name
        self.retry_times = retry_times

    def extend(self, prompt, system_prompt, seed=-1, *args, **kwargs):
        messages = [{
            'role': 'system',
            'content': system_prompt
        }, {
            'role': 'user',
            'content': prompt
        }]

        exception = None
        for _ in range(self.retry_times):
            try:
                response = dashscope.Generation.call(
                    self.model,
                    messages=messages,
                    seed=seed,
                    result_format='message',  # set the result to be "message" format.
                )
                assert response.status_code == HTTPStatus.OK, response
                expanded_prompt = response['output']['choices'][0]['message'][
                    'content']
                return PromptOutput(
                    status=True,
                    prompt=expanded_prompt,
                    seed=seed,
                    system_prompt=system_prompt,
                    message=json.dumps(response, ensure_ascii=False))
            except Exception as e:
                exception = e
        return PromptOutput(
            status=False,
            prompt=prompt,
            seed=seed,
            system_prompt=system_prompt,
            message=str(exception))

    def extend_with_img(self,
                        prompt,
                        system_prompt,
                        image: Union[Image.Image, str] = None,
                        seed=-1,
                        *args,
                        **kwargs):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        w = image.width
        h = image.height
        area = min(w * h, self.max_image_size)
        aspect_ratio = h / w
        resized_h = round(math.sqrt(area * aspect_ratio))
        resized_w = round(math.sqrt(area / aspect_ratio))
        image = image.resize((resized_w, resized_h))
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            image.save(f.name)
            fname = f.name
            image_path = f"file://{f.name}"
        prompt = f"{prompt}"
        messages = [
            {
                'role': 'system',
                'content': [{
                    "text": system_prompt
                }]
            },
            {
                'role': 'user',
                'content': [{
                    "text": prompt
                }, {
                    "image": image_path
                }]
            },
        ]
        response = None
        result_prompt = prompt
        exception = None
        status = False
        for _ in range(self.retry_times):
            try:
                response = dashscope.MultiModalConversation.call(
                    self.model,
                    messages=messages,
                    seed=seed,
                    result_format='message',  # set the result to be "message" format.
                )
                assert response.status_code == HTTPStatus.OK, response
                result_prompt = response['output']['choices'][0]['message'][
                    'content'][0]['text'].replace('\n', '\\n')
                status = True
                break
            except Exception as e:
                exception = e
        result_prompt = result_prompt.replace('\n', '\\n')
        os.remove(fname)

        return PromptOutput(
            status=status,
            prompt=result_prompt,
            seed=seed,
            system_prompt=system_prompt,
            message=str(exception) if not status else json.dumps(
                response, ensure_ascii=False))
