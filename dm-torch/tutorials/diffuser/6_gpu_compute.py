import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_memory = torch.cuda.get_device_properties(device).total_memory
    gpu_memory_available = gpu_memory - torch.cuda.memory_allocated(device)
    print(f"当前可用的GPU内存: {gpu_memory_available / 1024 ** 3} GB")
else:
    print("未检测到可用的GPU")
