import torch
from diffusers import StableDiffusionPipeline
from lycoris.kohya_model_utils import (
    load_file
)
from lycoris.utils import merge

from inner.tools.image_tools import show_img

TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"

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
        show_img(image)
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
        show_img(image)
