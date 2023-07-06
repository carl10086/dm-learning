import torch
import warnings

gpu_ok = False
# 检查CUDA是否可用
if torch.cuda.is_available():
    # 获取设备的计算能力
    device_cap = torch.cuda.get_device_capability()
    print(device_cap)
    # 检查设备的计算能力是否符合 (7, 0), (8, 0), 或 (9, 0) 中的一个
    if device_cap in ((7, 0), (8, 0), (9, 0)):
        gpu_ok = True

# 如果GPU不是 NVIDIA V100, A100, 或 H100, 则发出警告
if not gpu_ok:
    warnings.warn(
        "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower"
        "than expected."
    )