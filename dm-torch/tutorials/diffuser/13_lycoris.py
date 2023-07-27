import gc

import torch
from diffusers import StableDiffusionPipeline
from lycoris.kohya_model_utils import (
    load_file
)
from lycoris.utils import merge
import copy
from PIL import Image
from inner.tools import gpu_tools
from inner.tools.image_tools import show_image

TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"


def test_perf():
    # before merged
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=768,
        height=768,
        num_inference_steps=40,
        num_images_per_prompt=4,
        generator=generator
    ).images
    for image in images:
        show_image(image)
    # after merged
    merged = (
        pipeline.text_encoder,
        pipeline.vae,
        pipeline.unet
    )
    # load_lora_weights(pipeline, model_file)
    merge(
        merged,
        lyco,
        scale=0.7,
        device="cuda"
    )
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=768,
        height=768,
        num_inference_steps=40,
        num_images_per_prompt=4,
        generator=generator
    ).images
    for image in images:
        show_image(image)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


if __name__ == '__main__':
    model_file = "/root/autodl-tmp/models/ui_locoris/Miniature-world-style.safetensors"
    lyco = load_file(model_file, device="cuda")
    print("load state dict success")
    # model_dir = "/root/autodl-tmp/output/rev_diff"
    model_dir = "/root/autodl-tmp/output/diffusers/models/Chilloutmix"
    pipeline = StableDiffusionPipeline.from_pretrained(model_dir,
                                                       torch_dtype=torch.float16).to("cuda")

    generator = torch.manual_seed(248372777)
    prompt = "mini\(ttp\), (8k, RAW photo, best quality, masterpiece:1.2), cyberpunk city, miniature, landscape, isometric, "
    negative_prompt = "(((blur))), (EasyNegative:1.2), ng_deepnegative_v1_75t, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), bad anatomy,(long hair:1.4),DeepNegative,(fat:1.2),facing away, looking away,tilted head,lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digi"
    # test_perf()

    for _ in range(0, 4):
        images = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=768,
            height=768,
            num_inference_steps=40,
            num_images_per_prompt=4,
            generator=generator
        ).images
        show_image(image_grid(images, 2, 2))

        # print(f" before deep copy {gpu_tools.gpu_available_as_mb_str()}")

        copy_unet = copy.deepcopy(pipeline.unet)
        copy_te = copy.deepcopy(pipeline.text_encoder)
        # copy_vae = copy.deepcopy(pipeline.vae)
        # print(f"after deep copy {gpu_tools.gpu_available_as_mb_str()}")

        merged = (
            copy_te,
            # 看了源码，没动 vae
            pipeline.vae,
            copy_unet
        )

        merge(
            merged,
            lyco,
            scale=0.7,
            device="cuda"
        )

        copy_pipe = StableDiffusionPipeline(
            vae=pipeline.vae,
            text_encoder=copy_te,
            tokenizer=pipeline.tokenizer,
            unet=copy_unet,
            scheduler=pipeline.scheduler,
            safety_checker=None,
            feature_extractor=pipeline.feature_extractor,
            requires_safety_checker=False
        )

        images = copy_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=768,
            height=768,
            num_inference_steps=40,
            num_images_per_prompt=4,
            generator=generator
        ).images
        show_image(image_grid(images, 2, 2))

        images = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=768,
            height=768,
            num_inference_steps=40,
            num_images_per_prompt=4,
            generator=generator
        ).images
        show_image(image_grid(images, 2, 2))

        print(f"before empty cache: {gpu_tools.gpu_available_as_mb_str()}")
        del copy_unet
        del copy_te
        gc.collect()
        gpu_tools.force_empty_cache_for_gpu()
        print(f"after empty cache: {gpu_tools.gpu_available_as_mb_str()}")
