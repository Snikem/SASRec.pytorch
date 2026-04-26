import math
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


def _grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def _clamp(x, minv, maxv):
    return torch.clamp(x, minv, maxv)


class LSQQuantizer(nn.Module):
    def __init__(self, bits: int, signed: bool, init_scale: float | None = None):
        super().__init__()
        self.bits = bits
        self.signed = signed

        if signed:
            self.qn = -(2 ** (bits - 1))
            self.qp = (2 ** (bits - 1)) - 1
        else:
            self.qn = 0
            self.qp = (2 ** bits) - 1

        self.s = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("_inited", torch.tensor(0, dtype=torch.uint8))
        self._init_scale = init_scale

    @torch.no_grad()
    def _init_from(self, x: torch.Tensor):
        if self._init_scale is not None:
            self.s.fill_(float(self._init_scale))
        else:
            s = 2.0 * x.abs().mean() / math.sqrt(self.qp)
            self.s.copy_(s.clamp_min(1e-8))
        self._inited.fill_(1)

    def forward(self, x: torch.Tensor):
        if self.bits >= 32:
            return x

        if self._inited.item() == 0:
            self._init_from(x)

        g = 1.0 / math.sqrt(x.numel() * self.qp)
        s = _grad_scale(self.s, g).clamp_min(1e-8)

        x_hat = x / s
        x_hat = _clamp(x_hat, self.qn, self.qp)
        x_bar = RoundSTE.apply(x_hat)
        x_q = x_bar * s
        return x_q

class QuantConv2d(nn.Module):
    def __init__(
        self,
        conv: nn.Conv2d,
        w_bits: int = 8,
        a_bits: int = 8,
        act_signed: bool = False,
    ):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None

        self.wq = LSQQuantizer(bits=w_bits, signed=True)
        self.aq = LSQQuantizer(bits=a_bits, signed=act_signed)

    def forward(self, x):
        x_q = self.aq(x)
        w_q = self.wq(self.weight)
        return F.conv2d(
            x_q, w_q, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )

@torch.no_grad()
def load_fp32_to_lsq(fp32_model, lsq_model):
    lsq_model.conv1.weight.copy_(fp32_model.conv1.weight)
    if fp32_model.conv1.bias is not None:
        lsq_model.conv1.bias.copy_(fp32_model.conv1.bias)

    lsq_model.conv2.weight.copy_(fp32_model.conv2.weight)
    if fp32_model.conv2.bias is not None:
        lsq_model.conv2.bias.copy_(fp32_model.conv2.bias)

    lsq_model.conv3.weight.copy_(fp32_model.conv3.weight)
    if fp32_model.conv3.bias is not None:
        lsq_model.conv3.bias.copy_(fp32_model.conv3.bias)

class NetLSQ(nn.Module):
    def __init__(self, upscale_factor, w_bits=8, a_bits=8, quantize_last=False, pretrained_fp32 = None):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        conv3 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))

        self.conv1 = QuantConv2d(conv1, w_bits=w_bits, a_bits=a_bits, act_signed=False)
        self.conv2 = QuantConv2d(conv2, w_bits=w_bits, a_bits=a_bits, act_signed=False)
        if quantize_last:
            self.conv3 = QuantConv2d(conv3, w_bits=w_bits, a_bits=a_bits, act_signed=False)
        else:
            self.conv3 = conv3
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        if pretrained_fp32 is not None:
            load_fp32_to_lsq(pretrained_fp32, self)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x
