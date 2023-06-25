import safetensors
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler, StableDiffusionPipeline
import torch
import os
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from inner.tools import gpu_tools
from inner.tools.image_tools import show_img, show_images

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:8001'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:8001'

SD_MODEL = "runwayml/stable-diffusion-v1-5"

if __name__ == '__main__':
    gpu_tools.print_gpu_mem()
    vae = AutoencoderKL.from_pretrained(SD_MODEL, subfolder="vae", torch_dtype=torch.float16).to("cuda")
    gpu_tools.print_gpu_mem()
    # CLIP tokenizer 用来 分词 把文字转为 token 向量
    tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL, subfolder="tokenizer", torch_dtype=torch.float16)
    gpu_tools.print_gpu_mem()
    # CLIP text 文本 embedding 模型
    text_encoder = CLIPTextModel.from_pretrained(SD_MODEL, subfolder="text_encoder", torch_dtype=torch.float16).to(
        "cuda")
    gpu_tools.print_gpu_mem()
    # UNet2DConditionModel 扩散模型
    # unet = UNet2DConditionModel.from_pretrained(SD_MODEL, subfolder="unet", torch_dtype=torch.float16).to("cuda")
    unet = UNet2DConditionModel.from_pretrained("/root/autodl-tmp/output/rev_diff", subfolder="unet",
                                                torch_dtype=torch.float16).to("cuda")
    gpu_tools.print_gpu_mem()
    # 调度器
    scheduler = UniPCMultistepScheduler.from_pretrained(SD_MODEL, subfolder="scheduler")
    gpu_tools.print_gpu_mem()
    imageProcessor = CLIPImageProcessor.from_pretrained(SD_MODEL, torch_dtype=torch.float16,
                                                        subfolder="feature_extractor")

    prompt = "(masterpiece),(best quality),(ultra-detailed), (full body:1.2), 1girl,chibi,cute, smile, open mouth, flower, outdoors, playing guitar, music, beret, holding guitar, jacket, blush, tree, :3, shirt, short hair, cherry blossoms, green headwear, blurry, brown hair, blush stickers, long sleeves, bangs, headphones, black hair, pink flower, (beautiful detailed face), (beautiful detailed eyes), <lora:blindbox_v1_mix:1>"
    negative_prompt = (
        "(low quality:1.3), (worst quality:1.3)")

    lora_path = "/root/autodl-tmp/models/ui_lora/blindbox.safetensors"

    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=imageProcessor,
        requires_safety_checker=False,
    )
    lora_dict = safetensors.torch.load_file(lora_path, device="cpu")
    pipeline.load_lora_weights(lora_dict)
    images = pipeline(prompt=prompt,
                      negative_prompt=negative_prompt,
                      width=512,
                      height=512,
                      num_inference_steps=25,
                      num_images_per_prompt=8,
                      # generator=torch.manual_seed(0)
                      ).images
    # show_images(images)
    for image in images:
        show_img(image)
