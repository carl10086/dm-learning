import os

import torch
import concurrent.futures
import time
import numpy as np
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from inner.tools.image_tools import show_img

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:8001'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:8001'


def print_gpu_mem():
    device = torch.device("cuda")
    gpu_memory = torch.cuda.get_device_properties(device).total_memory
    gpu_memory_available = gpu_memory - torch.cuda.memory_allocated(device)
    print(f"当前可用的GPU内存: {gpu_memory_available / 1024 ** 3} GB")


print_gpu_mem()
pipeline = StableDiffusionPipeline.from_pretrained("/root/autodl-tmp/output/rev_diff", torch_dtype=torch.float16,
                                                   safety_checker=None).to("cuda")
# pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config, use_karras_sigmas=True
)
# 这个代码是强制关闭 flash attention 技术
# pipeline.unet.set_default_attn_processor()
# pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(
#     "cuda")
#

# optional use torch2 compile feature
# tomesd.apply_patch(pipeline.unet, ratio=0.75)


# pipeline.enable_attention_slicing()
# this can load c lora
# lora_path = "/root/autodl-tmp/models/ui_lora/revPopmart_v20.safetensors"
lora_path = "/root/autodl-tmp/models/ui_lora/blindbox.safetensors"
pipeline.load_lora_weights(lora_path)
# pipeline.unet.load_attn_procs(lora_path)
print_gpu_mem()


# pipeline.load_lora_weights("/root/autodl-tmp/models/ui_lora/blindbox.safetensors")


def time_wrapper(func):
    start_time = time.time()
    result = func()
    end_time = time.time()
    return result, end_time - start_time


def stress_test_v2(func, num_requests=100, num_workers=1):
    response_times = []
    successes = 0
    failures = 0
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for _ in range(num_requests):
            future = executor.submit(time_wrapper, func)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                result, response_time = future.result()

                if result:
                    response_times.append(response_time)
                    successes += 1
                else:
                    failures += 1

            except Exception as exc:
                print(f"Function {func.__name__} with argument  raised an exception: {exc}")
                failures += 1

    end_time = time.time()
    total_time = end_time - start_time
    qps = successes / total_time
    print(f"QPS: {qps:.4f} req/sec")

    response_times = np.array(response_times)
    print(f"Max response time: {np.max(response_times):.4f} sec")
    print(f"Min response time: {np.min(response_times):.4f} sec")
    print(f"Average response time: {np.mean(response_times):.4f} sec")
    print(f"P99 response time: {np.percentile(response_times, 99):.4f} sec")

    print(f"Number of successful requests: {successes}")
    print(f"Number of failed requests: {failures}")


def text_2_image():
    prompt = "(masterpiece),(best quality),(ultra-detailed), (full body:1.2), 1girl,chibi,cute, smile, open mouth, flower, outdoors, playing guitar, music, beret, holding guitar, jacket, blush, tree, :3, shirt, short hair, cherry blossoms, green headwear, blurry, brown hair, blush stickers, long sleeves, bangs, headphones, black hair, pink flower, (beautiful detailed face), (beautiful detailed eyes), <lora:blindbox_v1_mix:1>"
    negative_prompt = (
        "(low quality:1.3), (worst quality:1.3)")

    images = pipeline(prompt=prompt,
                      negative_prompt=negative_prompt,
                      width=512,
                      height=512,
                      num_inference_steps=25,
                      num_images_per_prompt=8,
                      # generator=torch.manual_seed(0)
                      ).images
    return images


if __name__ == '__main__':
    # 预热
    for image in text_2_image():
        show_img(image)
    # stress_test_v2(func=text_2_image, num_requests=10, num_workers=1)
