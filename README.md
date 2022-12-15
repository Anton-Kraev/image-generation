# image-generation
This mini-app uses Stable Diffusion pipelines for several tasks of conditional image generation: text2img, img2img, inpainting.

## Usage
Installing dependencies first:
```
pip install -r requirements.txt
```

Then run application:
```
streamlit run src/main.py
```

## References
[Model](https://huggingface.co/CompVis/stable-diffusion-v1-4)
[Pipeline](https://github.com/huggingface/diffusers/blob/main/examples/community/stable_diffusion_mega.py)