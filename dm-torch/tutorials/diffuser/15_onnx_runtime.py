import onnxruntime
from optimum.onnxruntime import ORTStableDiffusionPipeline
from diffusers import StableDiffusionPipeline
import torch

if __name__ == '__main__':
    # stable_diffusion = ORTStableDiffusionPipeline.from_pretrained("/root/autodl-tmp/output/rev_diff",
    #                                                               provider="CUDAExecutionProvider",
    #                                                               export=True)
    #
    # save_directory = "/root/autodl-tmp/output/diffusers/onnx/rev_diff"
    # stable_diffusion.save_pretrained(save_directory)
    #
    # session_options = onnxruntime.SessionOptions()
    # session_options.log_severity_level = 0
    # session_options = onnxruntime.SessionOptions()
    # session_options.log_severity_level = 0
    # model_id = "sd_v15_onnx"
    pipe = ORTStableDiffusionPipeline.from_pretrained(
        "/root/autodl-tmp/output/diffusers/onnx/sd_v15_onnx",
        # use_io_binding=True
    )
    pipe = pipe.to("cuda")
    # stable_diffusion = ORTStableDiffusionPipeline.from_pretrained(
    #     "/root/autodl-tmp/output/diffusers/onnx/rev_diff",
    #     provider="CUDAExecutionProvider",
    #     session_options=session_options
    # )
    # stable_diffusion = stable_diffusion.to("cuda")
    prompt = "sailing ship in storm by Leonardo da Vinci"
    for i in range(0, 10):
        images = pipe(
            prompt
        ).images
    #
    # pipe = StableDiffusionPipeline.from_pretrained("/root/autodl-tmp/output/rev_diff", torch_dtype=torch.float16,
    #                                                safety_checker=None).to("cuda")
    # for i in range(0, 10):
    #     images = pipe(
    #         prompt
    #     ).images
