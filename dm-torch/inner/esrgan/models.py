from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
import torchvision.transforms as transforms
from PIL import Image

from inner.tools import gpu_tools, image_tools


def analysis_model():
    print(torch.cuda.memory_allocated() / 1024 ** 2)
    total_params = sum(p.numel() for p in model.parameters())
    gpu_tools.force_empty_cache_for_gpu(gpu_device=device)
    print(f"Total number of parameters: {total_params}")
    print(gpu_tools.gpu_available_as_mb_str(device))
    del loadnet
    from torchsummary import summary
    summary(model, input_size=(1, 3, 512, 512))


def read_images_to_tensor(file_paths):
    images = []
    transform = transforms.ToTensor()

    for file_path in file_paths:
        img = Image.open(file_path).convert("RGB")
        img_tensor = transform(img).half()  # 将数据类型设置为float16
        images.append(img_tensor)

    # 将图像张量堆叠为一个四维张量
    tensor = torch.stack(images)

    return tensor


if __name__ == '__main__':
    device = "cuda:0"
    model_path = "/root/autodl-tmp/models/esrgan/RealESRGAN_x4plus_anime_6B.pth"
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    print(gpu_tools.gpu_available_as_mb_str(device))
    loadnet = torch.load(model_path, map_location=torch.device("cpu"))

    print(gpu_tools.gpu_available_as_mb_str(device))

    # prefer to use params_ema
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'

    model.load_state_dict(loadnet[keyname], strict=True)
    print(gpu_tools.gpu_available_as_mb_str(device))
    model.eval()
    model = model.half()
    model.to(device)

    x = read_images_to_tensor(
        [
            "/tmp/carl/output/1.jpeg",
            "/tmp/carl/output/2.jpeg",
            "/tmp/carl/output/3.jpeg",
            "/tmp/carl/output/4.jpeg",
        ]
    )

    x = x.to(device)
    print(f"after x to gpu {gpu_tools.gpu_available_as_mb_str(device)}")
    y = model(x)
    print(y.shape)
    # to_pil = transforms.ToPILImage()
    # image = to_pil(y.squeeze(0))
    # image_tools.show_img(image)
    torch.cuda.empty_cache()
    memory_allocated = torch.cuda.memory_allocated(device=device)
    print(f"Memory allocated during inference: {memory_allocated / (1024 * 1024)} MiB")
    print(f"after y to cpu {gpu_tools.gpu_available_as_mb_str(device)}")

    model.to("cpu")
    del model
    torch.cuda.empty_cache()
    print(f"after model to cpu {gpu_tools.gpu_available_as_mb_str(device)}")
