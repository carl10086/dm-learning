import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from inner.tools.image_tools import show_image

pipeline = StableDiffusionPipeline.from_ckpt(
    "/root/autodl-tmp/models/ui_models_1/rev_animated/Rev_Animated_v1.2.2_Pruned.safetensors",
    torch_dtype=torch.float16,
).to("cuda")


def text_2_image():
    prompt = "((solo)), ((1girl)), best quality, ultra high res, 8K, masterpiece, sharp focus,  clear gaze,popmart"
    negative_prompt = (
        "nsfw, (nipples:1.5), skin spots, acnes, skin blemishes, age spot, ugly, bad_anatomy, bad_hands, unclear fingers, twisted hands, fused fingers, fused legs, extra_hands, missing_fingers, broken hand, (more than two hands), well proportioned hands, more than two legs, unclear eyes, missing_arms, mutilated, extra limbs, extra legs, cloned face, extra_digit, fewer_digits, jpeg_artifacts,signature, watermark, username, blurry, mirror image, (worst quality:2), (low quality:2), (normal quality:2), lowres, ((monochrome)), ((grayscale)),((big hands, un-detailed skin, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime)), ((ugly mouth, ugly eyes, missing teeth, crooked teeth, close up, cropped, out of frame))")

    images = pipeline(prompt=prompt,
                      negative_prompt=negative_prompt,
                      width=512,
                      height=512,
                      num_inference_steps=20,
                      num_images_per_prompt=4,
                      # generator=torch.manual_seed(0)
                      ).images
    return images


if __name__ == '__main__':
    for image in text_2_image():
        show_image(image)
