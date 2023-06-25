import safetensors
import torch
from diffusers import StableDiffusionPipeline

from lycoris.kohya_model_utils import (
    load_models_from_stable_diffusion_checkpoint,
    save_stable_diffusion_checkpoint,
    load_file
)


from inner.tools.image_tools import show_img

TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"


def _convert_kohya_lora_to_diffusers(state_dict):
    unet_state_dict = {}
    te_state_dict = {}
    network_alpha = None

    for key, value in state_dict.items():
        if "lora_down" in key:
            lora_name = key.split(".")[0]
            lora_name_up = lora_name + ".lora_up.weight"
            lora_name_alpha = lora_name + ".alpha"
            if lora_name_alpha in state_dict:
                alpha = state_dict[lora_name_alpha].item()
                if network_alpha is None:
                    network_alpha = alpha
                    print(f"alpha:{alpha}")
                # elif network_alpha != alpha:
                #     raise ValueError("Network alpha is not consistent")

            if lora_name.startswith("lora_unet_"):
                diffusers_name = key.replace("lora_unet_", "").replace("_", ".")
                diffusers_name = diffusers_name.replace("down.blocks", "down_blocks")
                diffusers_name = diffusers_name.replace("mid.block", "mid_block")
                diffusers_name = diffusers_name.replace("up.blocks", "up_blocks")
                diffusers_name = diffusers_name.replace("transformer.blocks", "transformer_blocks")
                diffusers_name = diffusers_name.replace("to.q.lora", "to_q_lora")
                diffusers_name = diffusers_name.replace("to.k.lora", "to_k_lora")
                diffusers_name = diffusers_name.replace("to.v.lora", "to_v_lora")
                diffusers_name = diffusers_name.replace("to.out.0.lora", "to_out_lora")
                if "transformer_blocks" in diffusers_name:
                    if "attn1" in diffusers_name or "attn2" in diffusers_name:
                        diffusers_name = diffusers_name.replace("attn1", "attn1.processor")
                        diffusers_name = diffusers_name.replace("attn2", "attn2.processor")
                        unet_state_dict[diffusers_name] = value
                        unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict[lora_name_up]
            elif lora_name.startswith("lora_te_"):
                diffusers_name = key.replace("lora_te_", "").replace("_", ".")
                diffusers_name = diffusers_name.replace("text.model", "text_model")
                diffusers_name = diffusers_name.replace("self.attn", "self_attn")
                diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
                diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
                diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
                diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
                if "self_attn" in diffusers_name:
                    te_state_dict[diffusers_name] = value
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict[lora_name_up]

    unet_state_dict = {f"{UNET_NAME}.{module_name}": params for module_name, params in unet_state_dict.items()}
    te_state_dict = {f"{TEXT_ENCODER_NAME}.{module_name}": params for module_name, params in te_state_dict.items()}
    new_state_dict = {**unet_state_dict, **te_state_dict}
    return new_state_dict, network_alpha


def load_lora_weights(pipeline: StableDiffusionPipeline, model_file: str):
    pipeline._lora_scale = 1.0
    #
    # if use_safetensors and not is_safetensors_available():
    #     raise ValueError(
    #         "`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetenstors"
    #     )
    #
    # allow_pickle = False
    # if use_safetensors is None:
    #     use_safetensors = is_safetensors_available()
    #     allow_pickle = True
    #
    user_agent = {
        "file_type": "attn_procs_weights",
        "framework": "pytorch",
    }

    state_dict = safetensors.torch.load_file(model_file, device="cpu")

    # Convert kohya-ss Style LoRA attn procs to diffusers attn procs
    network_alpha = None
    if all((k.startswith("lora_te_") or k.startswith("lora_unet_")) for k in state_dict.keys()):
        state_dict, network_alpha = _convert_kohya_lora_to_diffusers(state_dict)

    # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
    # then the `state_dict` keys should have `self.unet_name` and/or `self.text_encoder_name` as
    # their prefixes.
    keys = list(state_dict.keys())
    if all(key.startswith(pipeline.unet_name) or key.startswith(pipeline.text_encoder_name) for key in keys):
        # Load the layers corresponding to UNet.
        unet_keys = [k for k in keys if k.startswith(pipeline.unet_name)]
        unet_lora_state_dict = {
            k.replace(f"{pipeline.unet_name}.", ""): v for k, v in state_dict.items() if k in unet_keys
        }
        # pipeline.unet.load_attn_procs(unet_lora_state_dict, network_alpha=network_alpha)
        pipeline.unet.load_attn_procs(unet_lora_state_dict)

        # Load the layers corresponding to text encoder and make necessary adjustments.
        text_encoder_keys = [k for k in keys if k.startswith(pipeline.text_encoder_name)]
        text_encoder_lora_state_dict = {
            k.replace(f"{pipeline.text_encoder_name}.", ""): v for k, v in state_dict.items() if k in text_encoder_keys
        }
        if len(text_encoder_lora_state_dict) > 0:
            attn_procs_text_encoder = pipeline._load_text_encoder_attn_procs(
                # text_encoder_lora_state_dict, network_alpha=network_alpha
                text_encoder_lora_state_dict
            )
            pipeline._modify_text_encoder(attn_procs_text_encoder)

            # save lora attn procs of text encoder so that it can be easily retrieved
            pipeline._text_encoder_lora_attn_procs = attn_procs_text_encoder

    # Otherwise, we're dealing with the old format. This means the `state_dict` should only
    # contain the module names of the `unet` as its keys WITHOUT any prefix.
    elif not all(
            key.startswith(pipeline.unet_name) or key.startswith(pipeline.text_encoder_name) for key in
            state_dict.keys()
    ):
        pipeline.unet.load_attn_procs(state_dict)
        warn_message = "You have saved the LoRA weights using the old format. To convert the old LoRA weights to the new format, you can first load them in a dictionary and then create a new dictionary like the following: `new_state_dict = {f'unet'.{module_name}: params for module_name, params in old_state_dict.items()}`."


if __name__ == '__main__':
    model_file = "/root/autodl-tmp/models/ui_locoris/nucleardiffv2.safetensors"
    # model_file = "/root/autodl-tmp/models/ui_locoris/MothCouture.safetensors"
    state_dict = safetensors.torch.load_file(model_file, device="cpu")
    # _convert_kohya_lora_to_diffusers(state_dict)
    for key, value in state_dict.items():
        # if "down" in key:
        print(key)

    print("load state dict success")
    pipeline = StableDiffusionPipeline.from_pretrained("/root/autodl-tmp/output/diffusers/models/Chilloutmix",
                                                       torch_dtype=torch.float16).to("cuda")
    # load_lora_weights(pipeline, model_file)
    pipeline.load_lora_weights(state_dict)
    # images = pipeline(
    #     prompt="mini\(ttp\), (8k, RAW photo, best quality, masterpiece:1.2), (realistic:1.37),(ancient temple), miniature, landscape, (isometric), glass tank,<lora:miniature_V1:0.6>, on table",
    #     negative_prompt="(((blur))),(EasyNegative:1.2), ng_deepnegative_v1_75t, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), bad anatomy,(long hair:1.4),DeepNegative,(fat:1.2),facing away, looking away,tilted head,lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digi",
    #     width=768,
    #     height=768,
    #     num_inference_steps=40,
    #     num_images_per_prompt=4,
    #     generator=torch.manual_seed(1736049741)
    # ).images
    # for image in images:
    #     show_img(image)

    # pipeline.save_lora_weights(
    #     save_directory='/root/autodl-tmp/output/diffusers/tmp',
    # unet_lora_layers = unet_lora_state_dict,
    # text_encoder_lora_layers = text_encoder_lora_state_dict,
    # unet_lora_layers=pipeline.unet.attn_processors,
    # text_encoder_lora_layers=pipeline.text_encoder_lora_attn_procs,
    # is_main_process=True,
    # weight_name="amixx_v20.ckpt"
    # )
