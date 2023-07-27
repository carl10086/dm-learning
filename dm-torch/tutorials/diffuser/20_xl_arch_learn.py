import torch
from diffusers import StableDiffusionPipeline

base_model_dir = "/root/autodl-tmp/output/rev_diff"
device = "cuda:0"

if __name__ == '__main__':
    pipe = StableDiffusionPipeline.from_pretrained(base_model_dir, torch_dtype=torch.float16)
    pipe.to(device)
    print(pipe.unet)
    pipe(
        prompt="hello world",
    )
