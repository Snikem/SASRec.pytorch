import torch
import torch.nn as nn
import torch.nn.functional as F


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
class PACTActivationQuantizer(nn.Module):
    def __init__(self, bits: int = 8, init_alpha: float = 6.0, eps: float = 1e-8):
        super().__init__()
        self.bits = bits
        self.eps = eps
        self.qn = 0
        self.qp = (2 ** bits) - 1

        self.alpha_param = nn.Parameter(torch.tensor(float(init_alpha)))

    def alpha(self):
        return F.softplus(self.alpha_param).clamp_min(self.eps)

    def forward(self, x: torch.Tensor):
        if self.bits >= 32:
            return x

        a = self.alpha()
        x = x.clamp_min(0.0)
        x = torch.min(x, a)

        scale = (a / float(self.qp)).clamp_min(self.eps)
        x_int = RoundSTE.apply(x / scale)
        x_int = torch.clamp(x_int, self.qn, self.qp)
        return x_int * scale

@torch.no_grad()
def copy_espcn_fp32_to_pact(fp32_model, pact_model):
    def copy_conv(src: nn.Conv2d, dst: nn.Conv2d, name: str):
        dst.weight.copy_(src.weight)
        if src.bias is not None:
            dst.bias.copy_(src.bias)

    copy_conv(fp32_model.conv1, pact_model.conv1, "conv1")
    copy_conv(fp32_model.conv2, pact_model.conv2, "conv2")
    copy_conv(fp32_model.conv3, pact_model.conv3, "conv3")


class NetPACTActivations(nn.Module):
    def __init__(self, upscale_factor, a_bits=8, init_alpha=6.0, in_channels=1, quantize_input=False, pretrained_fp32 = None):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2, bias=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self.q_in = None
        if quantize_input:
            self.q_in = PACTActivationQuantizer(bits=a_bits, init_alpha=init_alpha)
        self.q1 = PACTActivationQuantizer(bits=a_bits, init_alpha=init_alpha)
        self.q2 = PACTActivationQuantizer(bits=a_bits, init_alpha=init_alpha)
        
        if pretrained_fp32 is not None:
            copy_espcn_fp32_to_pact(pretrained_fp32, self)


    def forward(self, x):
        if self.q_in is not None:
            x = self.q_in(x)

        x = self.relu(self.conv1(x))
        x = self.q1(x)

        x = self.relu(self.conv2(x))
        x = self.q2(x)

        x = self.pixel_shuffle(self.conv3(x))
        return x
