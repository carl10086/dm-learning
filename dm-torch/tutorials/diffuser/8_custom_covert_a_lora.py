import safetensors
import torch
from diffusers import StableDiffusionPipeline
from diffusers.loaders import LoraLoaderMixin

if __name__ == '__main__':
    model_file = '/root/autodl-tmp/models/ui_lora/blindbox.safetensors'
    pipe = StableDiffusionPipeline.from_pretrained("/root/autodl-tmp/output/rev_diff", torch_dtype=torch.float16,
                                                   safety_checker=None)
    # mix_in = LoraLoaderMixin()
    #
    # state_dict = safetensors.torch.load_file(model_file, device="cpu")
    # # 转换LoRA attn procs
    # network_alpha = None
    # if all((k.startswith("lora_te_") or k.startswith("lora_unet_")) for k in state_dict.keys()):
    #     state_dict, network_alpha = pipe._convert_kohya_lora_to_diffusers(state_dict)
    #
    # # 提取权重
    # keys = list(state_dict.keys())
    # unet_lora_state_dict = None
    # text_encoder_lora_state_dict = None
    # if all(key.startswith(pipe.unet_name) or key.startswith(pipe.text_encoder_name) for key in keys):
    #     # 提取UNet的LoRA权重
    #     unet_keys = [k for k in keys if k.startswith(pipe.unet_name)]
    #     unet_lora_state_dict = {
    #         k.replace(f"{pipe.unet_name}.", ""): v for k, v in state_dict.items() if k in unet_keys
    #     }
    #
    #     # 提取text encoder的LoRA权重
    #     text_encoder_keys = [k for k in keys if k.startswith(pipe.text_encoder_name)]
    #     text_encoder_lora_state_dict = {
    #         k.replace(f"{pipe.text_encoder_name}.", ""): v for k, v in state_dict.items() if k in text_encoder_keys
    #     }
    #     attn_procs_text_encoder = pipe._load_text_encoder_attn_procs(
    #         text_encoder_lora_state_dict, network_alpha=network_alpha
    #     )


    pipe.load_lora_weights(model_file)

    pipe.save_lora_weights(
        save_directory='/root/autodl-tmp/output/diffusers/tmp',
        # unet_lora_layers=unet_lora_state_dict,
        # text_encoder_lora_layers=text_encoder_lora_state_dict,
        unet_lora_layers=pipe.unet.attn_processors,
        text_encoder_lora_layers=pipe.text_encoder_lora_attn_procs,
        is_main_process=True,
        weight_name="blindbox.ckpt"
    )
    # weight_name=weight_name
