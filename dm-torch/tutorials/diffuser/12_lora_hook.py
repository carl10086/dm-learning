import math

import numpy as np
import safetensors
import torch
import torch.nn as nn
import os
from PIL import Image

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline, \
    StableDiffusionLatentUpscalePipeline

from inner.tools.image_tools import show_images

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:8001'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:8001'


# modified from https://github.com/kohya-ss/sd-scripts/blob/ad5f318d066c52e5b27306b399bc87e41f2eef2b/networks/lora.py#L17
class LoRAModule(torch.nn.Module):
    def __init__(self, org_module: torch.nn.Module, lora_dim=4, alpha=1.0, multiplier=1.0):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()

        if isinstance(org_module, nn.Conv2d):
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim

        if isinstance(org_module, nn.Conv2d):
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if alpha is None or alpha == 0:
            self.alpha = self.lora_dim
        else:
            if type(alpha) == torch.Tensor:
                alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
            self.register_buffer("alpha", torch.tensor(alpha))  # Treatable as a constant.

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier

    def forward(self, x):
        scale = self.alpha / self.lora_dim
        return self.multiplier * scale * self.lora_up(self.lora_down(x))


class LoRAModuleContainer(torch.nn.Module):
    def __init__(self, hooks, state_dict, multiplier):
        super().__init__()
        self.multiplier = multiplier

        # Create LoRAModule from state_dict information
        for key, value in state_dict.items():
            if "lora_down" in key:
                lora_name = key.split(".")[0]
                lora_dim = value.size()[0]
                lora_name_alpha = key.split(".")[0] + ".alpha"
                alpha = None
                if lora_name_alpha in state_dict:
                    alpha = state_dict[lora_name_alpha].item()
                hook = hooks[lora_name]
                lora_module = LoRAModule(hook.orig_module, lora_dim=lora_dim, alpha=alpha, multiplier=multiplier)
                self.register_module(lora_name, lora_module)

        # Load whole LoRA weights
        self.load_state_dict(state_dict)

        # Register LoRAModule to LoRAHook
        for name, module in self.named_modules():
            if module.__class__.__name__ == "LoRAModule":
                hook = hooks[name]
                hook.append_lora(module)

    @property
    def alpha(self):
        return self.multiplier

    @alpha.setter
    def alpha(self, multiplier):
        self.multiplier = multiplier
        for name, module in self.named_modules():
            if module.__class__.__name__ == "LoRAModule":
                module.multiplier = multiplier

    def remove_from_hooks(self, hooks):
        for name, module in self.named_modules():
            if module.__class__.__name__ == "LoRAModule":
                hook = hooks[name]
                hook.remove_lora(module)
                del module


class LoRAHook(torch.nn.Module):
    """
    replaces forward method of the original Linear,
    instead of replacing the original Linear module.
    """

    def __init__(self):
        super().__init__()
        self.lora_modules = []

    def install(self, orig_module):
        assert not hasattr(self, "orig_module")
        self.orig_module = orig_module
        self.orig_forward = self.orig_module.forward
        self.orig_module.forward = self.forward

    def uninstall(self):
        assert hasattr(self, "orig_module")
        self.orig_module.forward = self.orig_forward
        del self.orig_forward
        del self.orig_module

    def append_lora(self, lora_module):
        self.lora_modules.append(lora_module)

    def remove_lora(self, lora_module):
        self.lora_modules.remove(lora_module)

    def forward(self, x):
        if len(self.lora_modules) == 0:
            return self.orig_forward(x)
        lora = torch.sum(torch.stack([lora(x) for lora in self.lora_modules]), dim=0)
        return self.orig_forward(x) + lora


class LoRAHookInjector(object):
    def __init__(self):
        super().__init__()
        self.hooks = {}
        self.device = None
        self.dtype = None

    def _get_target_modules(self, root_module, prefix, target_replace_modules):
        target_modules = []
        for name, module in root_module.named_modules():
            if (
                    module.__class__.__name__ in target_replace_modules and "transformer_blocks" not in name
            ):  # to adapt latest diffusers:
                for child_name, child_module in module.named_modules():
                    is_linear = isinstance(child_module, nn.Linear)
                    is_conv2d = isinstance(child_module, nn.Conv2d)
                    if is_linear or is_conv2d:
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        target_modules.append((lora_name, child_module))
        return target_modules

    def install_hooks(self, pipe):
        """Install LoRAHook to the pipe."""
        assert len(self.hooks) == 0
        text_encoder_targets = self._get_target_modules(pipe.text_encoder, "lora_te", ["CLIPAttention", "CLIPMLP"])
        unet_targets = self._get_target_modules(pipe.unet, "lora_unet", ["Transformer2DModel", "Attention"])
        for name, target_module in text_encoder_targets + unet_targets:
            hook = LoRAHook()
            hook.install(target_module)
            self.hooks[name] = hook
            # print(name)

        self.device = pipe.device
        self.dtype = pipe.unet.dtype

    def uninstall_hooks(self):
        """Uninstall LoRAHook from the pipe."""
        for k, v in self.hooks.items():
            v.uninstall()
        self.hooks = {}

    def apply_lora(self, filename, alpha=1.0):
        """Load LoRA weights and apply LoRA to the pipe."""
        assert len(self.hooks) != 0
        state_dict = safetensors.torch.load_file(filename)
        container = LoRAModuleContainer(self.hooks, state_dict, alpha)
        container.to(self.device, self.dtype)
        return container

    def remove_lora(self, container):
        """Remove the individual LoRA from the pipe."""
        container.remove_from_hooks(self.hooks)


def install_lora_hook(pipe: DiffusionPipeline):
    """Install LoRAHook to the pipe."""
    assert not hasattr(pipe, "lora_injector")
    assert not hasattr(pipe, "apply_lora")
    assert not hasattr(pipe, "remove_lora")
    injector = LoRAHookInjector()
    injector.install_hooks(pipe)
    pipe.lora_injector = injector
    pipe.apply_lora = injector.apply_lora
    pipe.remove_lora = injector.remove_lora


def uninstall_lora_hook(pipe: DiffusionPipeline):
    """Uninstall LoRAHook from the pipe."""
    pipe.lora_injector.uninstall_hooks()
    del pipe.lora_injector
    del pipe.apply_lora
    del pipe.remove_lora


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid


if __name__ == "__main__":
    torch.cuda.reset_peak_memory_stats()

    # pipe = StableDiffusionPipeline.from_pretrained(
    #     "gsdf/Counterfeit-V2.5", torch_dtype=torch.float16, safety_checker=None
    # ).to("cuda")

    # model_dir = "/root/autodl-tmp/output/diffusers/models/Chilloutmix"
    model_dir = "/root/autodl-tmp/output/rev_diff"
    pipe = StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float16,
                                                   safety_checker=None).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    # pipe.enable_xformers_memory_efficient_attention()
    #
    # prompt = "masterpeace, best quality, highres, 1girl, at dusk"
    # negative_prompt = (
    #     "(low quality, worst quality:1.4), (bad anatomy), (inaccurate limb:1.2), "
    #     "bad composition, inaccurate eyes, extra digit, fewer digits, (extra arms:1.2) "
    # )

    # prompt = "(masterpiece),(best quality),(ultra-detailed), (full body:1.2), 1girl,chibi,cute, smile, open mouth, flower, outdoors, playing guitar, music, beret, holding guitar, jacket, blush, tree, :3, shirt, short hair, cherry blossoms, green headwear, blurry, brown hair, blush stickers, long sleeves, bangs, headphones, black hair, pink flower, (beautiful detailed face), (beautiful detailed eyes), <lora:blindbox_v1_mix:1>"
    # negative_prompt = (
    #     "(low quality:1.3), (worst quality:1.3)")
    # lora_fn = "../stable-diffusion-study/models/lora/light_and_shadow.safetensors"
    prompt = "mini\(ttp\), (8k, RAW photo, best quality, masterpiece:1.2), (realistic:1.37),(ancient temple), miniature, landscape, (isometric), glass tank,<lora:miniature_V1:0.6>, on table",
    negative_prompt = "(((blur))),(EasyNegative:1.2), ng_deepnegative_v1_75t, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), bad anatomy,(long hair:1.4),DeepNegative,(fat:1.2),facing away, looking away,tilted head,lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digi",
    lora_fn = "/root/autodl-tmp/models/ui_locoris/nucleardiffv2.safetensors"

    # Without Lora
    # images = pipe(
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     width=512,
    #     height=768,
    #     num_inference_steps=15,
    #     num_images_per_prompt=4,
    #     generator=torch.manual_seed(941189224),
    # ).images
    # image_grid(images, 1, 4).save("test_orig.png")
    # show_images(images)

    mem_bytes = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    print(f"Without Lora -> {mem_bytes / (10 ** 6)}MB")

    # Hook version (some restricted apply)
    generator = torch.manual_seed(0)
    install_lora_hook(pipe)
    pipe.apply_lora(lora_fn)
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=768,
        height=768,
        num_inference_steps=30,
        num_images_per_prompt=4,
        generator=generator,
        # output_type="latent"
    ).images

    # upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler",
    #                                                                 torch_dtype=torch.float16)
    # upscaler.to("cuda")
    #
    # images = upscaler(
    #     prompt=prompt,
    #     image=low_res_latents,
    #     num_inference_steps=20,
    #     guidance_scale=0,
    #     generator=generator,
    # ).images

    # image_grid(images, 1, 4).save("test_lora_hook.png")
    # image_grid(images, 1, 4)
    show_images(images)
    i = 0
    # for image in images:
    #     i = i + 1
    #     image.save(f"/tmp/carl/output/{i}.jpeg")
    uninstall_lora_hook(pipe)

    mem_bytes = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    print(f"Hook version -> {mem_bytes / (10 ** 6)}MB")

    # Diffusers dev version
    # pipe.load_lora_weights(lora_fn)
    # images = pipe(
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     width=512,
    #     height=768,
    #     num_inference_steps=15,
    #     num_images_per_prompt=4,
    #     generator=torch.manual_seed(0),
    #     # cross_attention_kwargs={"scale": 0.5},  # lora scale
    # ).images
    # # image_grid(images, 1, 4).save("test_lora_dev.png")
    # image_grid(images, 1, 4)
    #
    # mem_bytes = torch.cuda.max_memory_allocated()
    # print(f"Diffusers dev version -> {mem_bytes / (10 ** 6)}MB")
    #
    # abs-difference image
    # image_hook = np.array(Image.open("test_lora_hook.png"), dtype=np.int16)
    # image_dev = np.array(Image.open("test_lora_dev.png"), dtype=np.int16)
    # image_diff = Image.fromarray(np.abs(image_hook - image_dev).astype(np.uint8))
    # image_diff.save("test_lora_hook_dev_diff.png")
