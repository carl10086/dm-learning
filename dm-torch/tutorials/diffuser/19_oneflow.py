from diffusers import DiffusionPipeline
import torch
import os

from inner.tools import image_tools

os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:8001'
if __name__ == '__main__':
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16,
                                             use_safetensors=True, variant="fp16")
    pipe.to("cuda")

    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()

    prompt = "An astronaut riding a green horse"

    images = pipe(prompt=prompt).images[0]

    image_tools.show_images(images)
