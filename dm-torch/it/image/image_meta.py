from PIL import Image
import hashlib

import numpy as np
from safetensors import safe_open
import torch


image_path = '/tmp/carl/00000-3121882917.png'

with open(image_path, 'rb') as f:
    image = Image.open(f)
    print(image.info)


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""

    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'




def check_file_format_and_precision(file_path):
    try:
        # 尝试使用 numpy 的 load 函数来加载文件
        data = np.load(file_path)
        dtype = data.dtype
        if dtype == np.float16:
            precision = "fp16"
        elif dtype == np.float32:
            precision = "fp32"
        else:
            precision = "unknown"
        return "numpy", precision
    except:
        pass

    try:
        # 尝试使用 safetensors 的 safe_open 函数来加载文件
        with safe_open(file_path, framework="pt", device="cpu") as f:
            # 获取第一个张量的数据类型
            first_key = list(f.keys())[0]
            tensor = f.get_tensor(first_key)
            dtype = tensor.dtype
            if dtype == torch.float16:
                precision = "fp16"
            elif dtype == torch.float32:
                precision = "fp32"
            else:
                precision = "unknown"
        return "safetensors", precision
    except:
        pass

    return "unknown", "unknown"


# print(model_hash('/root/autodl-tmp/models/ui_models_1/rev_animated/Rev_Animated_v1.2.2_Pruned.safetensors'))
# print(calculate_sha256('/root/autodl-tmp/models/ui_models_1/rev_animated/Rev_Animated_v1.2.2_Pruned.safetensors'))
print(check_file_format_and_precision('/root/autodl-tmp/models/ui_models_1/rev_animated/Rev_Animated_v1.2.2_Pruned.safetensors'))
