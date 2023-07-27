import os

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler, StableDiffusionPipeline
# 这里假设`safetensors.torch`是一个库，你可以根据实际情况导入相应的库
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from inner.tools import gpu_tools

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:8001'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:8001'

SD_MODEL = "runwayml/stable-diffusion-v1-5"


class LoraCache:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LoraCache, cls).__new__(cls, *args, **kwargs)
            cls._instance.cache = {}
        return cls._instance

    def load_lora(self, name, lora_path, device="cpu"):
        """
        加载lora文件并以易读名称缓存到字典中。
        :param name: 易读名称作为缓存的键
        :param lora_path: Lora文件路径
        :param device: 设备类型, 默认为"cpu"
        """
        # 如果名称已经在缓存中，直接返回缓存的数据
        if name in self.cache:
            return self.cache[name]

        # 否则，加载lora文件
        try:
            # 这里假设`safetensors.torch`是一个库，你可以根据实际情况导入相应的库
            import safetensors.torch
            lora_dict = safetensors.torch.load_file(lora_path, device=device)
        except ImportError:
            raise ImportError("需要safetensors库来加载lora文件.")
        except Exception as e:
            raise Exception(f"加载Lora文件失败: {e}")

        # 将lora字典以易读名称存储在缓存中
        self.cache[name] = lora_dict

        # 返回lora字典
        return lora_dict

    def clear_cache(self):
        """
        清空缓存。
        """
        self.cache.clear()

    def get_lora_by_name(self, name):
        """
        通过易读名称获取缓存的lora字典。
        :param name: 易读名称
        :return: lora字典
        """
        return self.cache.get(name)


vae = AutoencoderKL.from_pretrained(SD_MODEL, subfolder="vae", torch_dtype=torch.float16).to("cuda")
# CLIP tokenizer 用来 分词 把文字转为 token 向量
tokenizer = CLIPTokenizer.from_pretrained(SD_MODEL, subfolder="tokenizer", torch_dtype=torch.float16)
# CLIP text 文本 embedding 模型
text_encoder = CLIPTextModel.from_pretrained(SD_MODEL, subfolder="text_encoder", torch_dtype=torch.float16).to(
    "cuda")
# UNet2DConditionModel 扩散模型
unet = UNet2DConditionModel.from_pretrained(SD_MODEL, subfolder="unet", torch_dtype=torch.float16).to("cuda")
# unet = torch.compile(
#     UNet2DConditionModel.from_pretrained(
#         "/root/autodl-tmp/output/rev_diff",
#         subfolder="unet",
#         torch_dtype=torch.float16).to("cuda"), mode="reduce-overhead",
#     fullgraph=True)
# 调度器
scheduler = UniPCMultistepScheduler.from_pretrained(SD_MODEL, subfolder="scheduler")
gpu_tools.print_gpu_mem()
imageProcessor = CLIPImageProcessor.from_pretrained(SD_MODEL, torch_dtype=torch.float16,
                                                    subfolder="feature_extractor")

print("加载了 unet")
global_lora_cache = LoraCache()
# global_lora_cache.load_lora("blindbox", "/root/autodl-tmp/models/ui_lora/blindbox.safetensors")
# global_lora_cache.load_lora("popmart", "/root/autodl-tmp/models/ui_lora/revPopmart_v20.safetensors")
# global_lora_cache.load_lora("3dmm", "/root/autodl-tmp/models/ui_lora/3DMM_V11.safetensors")
print("加载了 2个lora")


def text_2_img(lora: str):
    prompt = "(masterpiece),(best quality),(ultra-detailed), (full body:1.2), 1girl,chibi,cute, smile, open mouth, flower, outdoors, playing guitar, music, beret, holding guitar, jacket, blush, tree, :3, shirt, short hair, cherry blossoms, green headwear, blurry, brown hair, blush stickers, long sleeves, bangs, headphones, black hair, pink flower, (beautiful detailed face), (beautiful detailed eyes), <lora:blindbox_v1_mix:1>"
    negative_prompt = (
        "(low quality:1.3), (worst quality:1.3)")
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
    # pipeline.load_lora_weights(global_lora_cache.get_lora_by_name(lora))
    pipeline(prompt=prompt,
             negative_prompt=negative_prompt,
             width=768,
             height=768,
             num_inference_steps=25,
             num_images_per_prompt=1,
             )
    print(f"使用lora:{lora} 推理之后")


if __name__ == '__main__':
    for i in range(10):
        text_2_img("blindbox")
        # text_2_img("popmart")
        # text_2_img("3dmm")
