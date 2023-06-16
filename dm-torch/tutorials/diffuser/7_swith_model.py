import os

import torch
import concurrent.futures
import time
import numpy as np
import gc
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:8001'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:8001'


def print_gpu_mem():
    device = torch.device("cuda")
    gpu_memory = torch.cuda.get_device_properties(device).total_memory
    gpu_memory_available = gpu_memory - torch.cuda.memory_allocated(device)
    print(f"当前可用的GPU内存: {gpu_memory_available / 1024 ** 3} GB")


def load_model_to_memory(path, dtype=torch.float16):
    # 加载模型到内存中，而不是GPU
    return StableDiffusionPipeline.from_pretrained(path, torch_dtype=dtype, map_location="cpu", safety_checker=None)


print_gpu_mem()
model_dict = {
    "1": load_model_to_memory("/root/autodl-tmp/output/rev_diff"),
    "2": load_model_to_memory("runwayml/stable-diffusion-v1-5")
}
print_gpu_mem()


def text_2_image(model_id):
    print("before loading model print gpu available memory")
    print_gpu_mem()
    prompt = "(masterpiece),(best quality),(ultra-detailed), (full body:1.2), 1girl,chibi,cute, smile, open mouth, flower, outdoors, playing guitar, music, beret, holding guitar, jacket, blush, tree, :3, shirt, short hair, cherry blossoms, green headwear, blurry, brown hair, blush stickers, long sleeves, bangs, headphones, black hair, pink flower, (beautiful detailed face), (beautiful detailed eyes), <lora:blindbox_v1_mix:1>"
    negative_prompt = "(low quality:1.3), (worst quality:1.3)"

    # 记录模型加载到 GPU 的时间
    start = time.time()
    model = model_dict[model_id].to("cuda")
    print("after loading model print gpu available memory")
    print_gpu_mem()
    loading_time = time.time() - start

    # 记录模型推理的时间
    start = time.time()
    images = model(prompt=prompt,
                   negative_prompt=negative_prompt,
                   width=512,
                   height=512,
                   num_inference_steps=25,
                   num_images_per_prompt=1,
                   ).images
    inference_time = time.time() - start

    # Move model back to CPU
    model.to("cpu")
    # 记录释放 GPU 资源的时间
    start = time.time()
    torch.cuda.empty_cache()

    del model
    # gc.collect()
    release_time = time.time() - start

    print(f"Model loading time: {loading_time} seconds")
    print(f"Inference time: {inference_time} seconds")
    print(f"Release time: {release_time} seconds")
    print("finish infer image, model can print gpu available memory")
    print_gpu_mem()
    return images
