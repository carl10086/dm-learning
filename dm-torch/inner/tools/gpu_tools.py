import torch


def print_gpu_mem():
    device = torch.device("cuda")
    gpu_memory = torch.cuda.get_device_properties(device).total_memory
    gpu_memory_available = gpu_memory - torch.cuda.memory_allocated(device)
    print(f"当前可用的GPU内存: {gpu_memory_available / 1024 ** 3} GB")


def gpu_available():
    device = torch.device("cuda")
    gpu_memory = torch.cuda.get_device_properties(device).total_memory
    gpu_memory_available = gpu_memory - torch.cuda.memory_allocated(device) / 1024 ** 2
    return gpu_memory_available


if __name__ == '__main__':
    print(gpu_available())
