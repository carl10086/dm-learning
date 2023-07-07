import os

import torch
from diffusers import StableDiffusionControlNetPipeline, AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler, \
    ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from diffusers.utils import load_image
import cv2
from PIL import Image
import numpy as np
from transformers import CLIPTextModel, CLIPImageProcessor, CLIPTokenizer
from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, \
    NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, \
    MediapipeFaceDetector, SamDetector, LeresDetector

from inner.tools import image_tools, gpu_tools

os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:8001'

device = "cuda:0"
base_model_dir = "/root/autodl-tmp/output/rev_diff"
repo_id = "lllyasviel/control_v11p_sd15_lineart"


def convert_2_canny(image: Image) -> Image:
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def do_infer_img2img():
    # repo_id = "lllyasviel/control_v11p_sd15_canny"
    controlnet = ControlNetModel.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline(
        vae=AutoencoderKL.from_pretrained(
            base_model_dir, subfolder="vae",
            torch_dtype=torch.float16
        ),
        text_encoder=CLIPTextModel.from_pretrained(base_model_dir,
                                                   subfolder="text_encoder",
                                                   torch_dtype=torch.float16),
        unet=UNet2DConditionModel.from_pretrained(
            base_model_dir, subfolder="unet", torch_dtype=torch.float16
        ),
        scheduler=UniPCMultistepScheduler.from_pretrained(base_model_dir,
                                                          subfolder="scheduler"
                                                          ),
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=CLIPImageProcessor.from_pretrained(
            base_model_dir,
            torch_dtype=torch.float16,
            subfolder="feature_extractor"),
        tokenizer=CLIPTokenizer.from_pretrained(
            base_model_dir,
            subfolder="tokenizer"
        ),
        controlnet=controlnet
    )

    pipe.to(device)
    prompt = ", best quality, extremely detailed"
    prompt = [t + prompt for t in
              ["Sandra Oh", "Kim Kardashian", "rihanna", "taylor swift"]]  # 分别为: 吴珊卓、金·卡戴珊、蕾哈娜、泰勒·斯威夫特
    generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]
    output = pipe(
        # prompt="futuristic-looking woman",
        prompt=prompt,
        image=[image] * 4,
        control_image=control_image,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
        num_inference_steps=20,
        generator=generator,
    )
    # image_tools.show_images(output.images)
    image_tools.show_image(image_grid(output.images, 2, 2))


def do_infer():
    # repo_id = "lllyasviel/control_v11p_sd15_canny"
    controlnet = ControlNetModel.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline(
        vae=AutoencoderKL.from_pretrained(
            base_model_dir, subfolder="vae",
            torch_dtype=torch.float16
        ),
        text_encoder=CLIPTextModel.from_pretrained(base_model_dir,
                                                   subfolder="text_encoder",
                                                   torch_dtype=torch.float16),
        unet=UNet2DConditionModel.from_pretrained(
            base_model_dir, subfolder="unet", torch_dtype=torch.float16
        ),
        scheduler=UniPCMultistepScheduler.from_pretrained(base_model_dir,
                                                          subfolder="scheduler"
                                                          ),
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=CLIPImageProcessor.from_pretrained(
            base_model_dir,
            torch_dtype=torch.float16,
            subfolder="feature_extractor"),
        tokenizer=CLIPTokenizer.from_pretrained(
            base_model_dir,
            subfolder="tokenizer"
        ),
        controlnet=controlnet
    )

    pipe.to(device)
    prompt = ", best quality, extremely detailed"
    prompt = [t + prompt for t in
              ["Sandra Oh", "Kim Kardashian", "rihanna", "taylor swift"]]  # 分别为: 吴珊卓、金·卡戴珊、蕾哈娜、泰勒·斯威夫特
    generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]
    output = pipe(
        prompt,
        control_image,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
        num_inference_steps=20,
        generator=generator,
    )
    image_tools.show_image(image_grid(output.images, 2, 2))


def print_gpu_available(fmt: str):
    print(f"{fmt} then gpu remain: {gpu_tools.gpu_available_as_mb_str('cuda:0')}")


if __name__ == '__main__':
    image = load_image("/tmp/carl/input/input.png")
    image_tools.show_image(image)
    lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
    # after_image = convert_2_canny(image)
    control_image = lineart(image)
    for _ in range(0, 3):
        do_infer()
        do_infer_img2img()
