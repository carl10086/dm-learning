import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def gpu_index(device: str) -> int:
    if ':' in device:
        return int(device.split(":")[-1])
    return 0


def gpu_available_as_mb_str(device: str) -> str:
    device = torch.device(device)
    gpu_memory = torch.cuda.get_device_properties(device).total_memory
    gpu_memory_reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    # gpu_memory_available = gpu_memory - gpu_memory_reserved
    gpu_memory_available = gpu_memory - allocated
    # gpu_memory_available = nvidia_gpu_availabl_as_bytes(gpu_index(device))
    return f"{gpu_memory_available / 1024 ** 2:.2f} MB"


def force_empty_cache_for_gpu(gpu_device: str):
    device_index = gpu_index(device=gpu_device)
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()


def nvidia_gpu_availabl_as_bytes(device_id: int) -> str:
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_id)
    info = nvmlDeviceGetMemoryInfo(handle)
    total_memory = info.total
    used_memory = info.used
    gpu_memory_available = total_memory - used_memory
    return gpu_memory_available
