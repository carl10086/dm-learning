import math
import torch
import torch.nn as nn
from diffusers import DiffusionPipeline


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
        # lora rank
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


if __name__ == '__main__':
    pass
