!pip install diffusers transformers tokenizers

from huggingface_hub import notebook_login
notebook_login()

import torch
assert torch.cuda.is_available()
!nvidia-smi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

assert torch.cuda.is_available()
!nvidia-smi

pip install accelerate

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    use_auth_token='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',
    variant="fp16", torch_dtype=torch.float16
).to("cuda")

prompt = "a bike in beach"
image = pipe(prompt).images[0]
image.save(f"bike.png")
image

generator = torch.Generator("cuda").manual_seed(1024)

prompt = "a bike in beach"
image = pipe(prompt, generator=generator).images[0]
image.save(f"bike1.png")
image

prompt = "a bike in beach"
image = pipe(prompt, num_inference_steps=25).images[0]
image.save(f"bike2.png")
image

prompt = "a bike in beach"
image = pipe(prompt, generator=generator,negative_prompt="tree").images[0]
image.save(f"bike3.png")
image
