import torch
from diffusers import StableDiffusionPipeline


class Model:
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            custom_pipeline='stable_diffusion_mega',
        )
        self.pipe.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipe.enable_attention_slicing()

    def generate(self, pipeline, params):
        prompt = params.get('prompt')
        image = params.get('init_img')
        mask_image = params.get('mask_img')
        negative_prompt = params.get('negative_prompt')
        guidance_scale = params.get('guidance_scale')
        num_inference_steps = params.get('num_inference_steps')
        num_images_per_prompt = params.get('num_images_per_prompt')
        if pipeline == 'text2img':
            return self.pipe.text2img(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale,
                                      num_inference_steps=num_inference_steps,
                                      num_images_per_prompt=num_images_per_prompt).images
        if pipeline == 'img2img':
            return self.pipe.img2img(prompt=prompt, image=image, negative_prompt=negative_prompt,
                                     guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                                     num_images_per_prompt=num_images_per_prompt).images
        if pipeline == 'inpaint':
            return self.pipe.inpaint(prompt=prompt, image=image, mask_image=mask_image, negative_prompt=negative_prompt,
                                     guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                                     num_images_per_prompt=num_images_per_prompt).images
