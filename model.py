from io import BytesIO

import torch
import requests
from PIL import Image

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "C:\\Users\\anton\\stable-diffusion-v1-4",
    custom_pipeline="stable_diffusion_mega",
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")
pipe.enable_attention_slicing()
