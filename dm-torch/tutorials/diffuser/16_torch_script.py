import functools
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, Tuple

import torch
from diffusers import StableDiffusionPipeline

from inner.tools.image_tools import show_images

# torch disable grad
torch.set_grad_enabled(False)

# set variables
n_experiments = 2
unet_runs_per_experiment = 50


# load inputs
def generate_inputs():
    sample = torch.randn(2, 4, 64, 64).half().cuda()
    timestep = torch.rand(1).half().cuda() * 999
    encoder_hidden_states = torch.randn(2, 77, 768).half().cuda()
    return sample, timestep, encoder_hidden_states


def jit_compile():
    global unet, unet
    pipe = StableDiffusionPipeline.from_pretrained(
        "/root/autodl-tmp/output/rev_diff",
        torch_dtype=torch.float16,
    ).to("cuda")
    unet = pipe.unet
    unet.eval()
    unet.to(memory_format=torch.channels_last)  # use channels_last memory format
    unet.forward = functools.partial(unet.forward, return_dict=False)  # set return_dict=False as default
    # warmup
    for _ in range(3):
        with torch.inference_mode():
            inputs = generate_inputs()
            orig_output = unet(*inputs)
    # trace
    print("tracing..")
    unet_traced = torch.jit.trace(unet, inputs)
    unet_traced.eval()
    print("done tracing")
    # warmup and optimize graph
    for _ in range(5):
        with torch.inference_mode():
            inputs = generate_inputs()
            orig_output = unet_traced(*inputs)
    # benchmarking
    with torch.inference_mode():
        for _ in range(n_experiments):
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(unet_runs_per_experiment):
                orig_output = unet_traced(*inputs)
            torch.cuda.synchronize()
            print(f"unet traced inference took {time.time() - start_time:.2f} seconds")
        for _ in range(n_experiments):
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(unet_runs_per_experiment):
                orig_output = unet(*inputs)
            torch.cuda.synchronize()
            print(f"unet inference took {time.time() - start_time:.2f} seconds")
    unet_traced.save("/root/autodl-tmp/output/diffusers/jit/unet_traced.pt")


@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor


# del pipe.unet
pipe = StableDiffusionPipeline.from_pretrained(
    "/root/autodl-tmp/output/rev_diff",
    torch_dtype=torch.float16,
).to("cuda")

# use jitted unet
unet_traced = torch.jit.load("/root/autodl-tmp/output/diffusers/jit/unet_traced.pt")


class TracedUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = pipe.unet.in_channels
        self.device = pipe.unet.device
        self.config = pipe.unet.config

    def forward(self, latent_model_input, t, encoder_hidden_states, **kwargs):
        sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
        return [sample]


if __name__ == '__main__':
    # pipe.unet = TracedUNet()
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    prompt = "(masterpiece),(best quality),(ultra-detailed), (full body:1.2), 1girl,chibi,cute, smile, open mouth, flower, outdoors, playing guitar, music, beret, holding guitar, jacket, blush, tree, :3, shirt, short hair, cherry blossoms, green headwear, blurry, brown hair, blush stickers, long sleeves, bangs, headphones, black hair, pink flower, (beautiful detailed face), (beautiful detailed eyes), <lora:blindbox_v1_mix:1>"
    negative_prompt = ("(low quality:1.3), (worst quality:1.3)")
    # with torch.inference_mode():
    # image = pipe([prompt] * 1, num_inference_steps=50).images[0]
    for i in range(0, 3):
        images = pipe(
            prompt,
            negative_prompt=negative_prompt
        ).images

        show_images(images)
