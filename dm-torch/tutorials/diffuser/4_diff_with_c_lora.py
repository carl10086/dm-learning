import torch
import concurrent.futures
import time
import numpy as np
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from inner.tools.image_tools import show_img

pipeline = StableDiffusionPipeline.from_pretrained("/root/autodl-tmp/output/rev_diff", torch_dtype=torch.float16,
                                                   safety_checker=None).to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config, use_karras_sigmas=True
)

# optional use torch2 compile feature
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

# pipeline.enable_attention_slicing()
# this can load c lora
pipeline.load_lora_weights("/root/autodl-tmp/models/ui_lora/revPopmart_v20.safetensors")


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
    prompt = "((solo)), ((1girl)), best quality, ultra high res, 8K, masterpiece, sharp focus,  clear gaze,popmart"
    negative_prompt = (
        "nsfw, (nipples:1.5), skin spots, acnes, skin blemishes, age spot, ugly, bad_anatomy, bad_hands, unclear fingers, twisted hands, fused fingers, fused legs, extra_hands, missing_fingers, broken hand, (more than two hands), well proportioned hands, more than two legs, unclear eyes, missing_arms, mutilated, extra limbs, extra legs, cloned face, extra_digit, fewer_digits, jpeg_artifacts,signature, watermark, username, blurry, mirror image, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)),((big hands, un-detailed skin, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime)), ((ugly mouth, ugly eyes, missing teeth, crooked teeth, close up, cropped, out of frame))")

    images = pipeline(prompt=prompt,
                      negative_prompt=negative_prompt,
                      width=512,
                      height=512,
                      num_inference_steps=20,
                      num_images_per_prompt=8,
                      # generator=torch.manual_seed(0)
                      ).images
    return images


if __name__ == '__main__':
    # images = text_2_image()
    for image in text_2_image():
        show_img(image)
    stress_test_v2(func=text_2_image, num_requests=10, num_workers=1)
