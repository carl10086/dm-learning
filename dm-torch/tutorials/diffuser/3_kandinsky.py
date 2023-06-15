from diffusers import DiffusionPipeline
import torch
import os

from inner.tools.image_tools import show_img

# os.environ['HTTP_PROXY'] = 'http://192.168.126.12:12798'
# os.environ['HTTPS_PROXY'] = 'http://192.168.126.12:12798'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:8001'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:8001'

# 1. use prior 把 文本 embedding -> image bedding
pipe_prior = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16)
pipe_prior.to("cuda")

t2i_pipe = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16)
t2i_pipe.to("cuda")


def simple_txt_2_img(prompt: str):
    # prompt = "A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting"
    # prompt = "An Asia man face, handsome, smile, long hair, big eyes"
    negative_prompt = "low quality, bad quality"

    # generator = torch.Generator(device="cuda").manual_seed(0)
    image_embeds, negative_image_embeds = pipe_prior(prompt, negative_prompt, guidance_scale=1.0,
                                                     # generator=generator
                                                     ).to_tuple()

    images = t2i_pipe(prompt,
                      negative_prompt=negative_prompt,
                      image_embeds=image_embeds,
                      num_images_per_prompt=4,
                      negative_image_embeds=negative_image_embeds).images
    # image.save("cheeseburger_monster.png")

    for image in images:
        show_img(image)


if __name__ == '__main__':
    simple_txt_2_img("A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting")
