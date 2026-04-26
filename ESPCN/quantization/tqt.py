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


class TQTQuantizer(nn.Module):
    def __init__(self, bits: int = 8, signed: bool = True, init_t: float | None = None, eps: float = 1e-8):
        super().__init__()
        self.bits = bits
        self.signed = signed
        self.eps = eps

        if signed:
            self.qn = -(2 ** (bits - 1))
            self.qp = (2 ** (bits - 1)) - 1
        else:
            self.qn = 0
            self.qp = (2 ** bits) - 1

        self.log_t = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("_inited", torch.tensor(0, dtype=torch.uint8))
        self._init_t = init_t

    @torch.no_grad()
    def _init_from(self, x: torch.Tensor):
        if self._init_t is not None:
            t0 = float(self._init_t)
        else:
            if self.signed:
                t0 = float(x.detach().abs().max().clamp_min(self.eps).item())
            else:
                t0 = float(x.detach().max().clamp_min(self.eps).item())

        self.log_t.copy_(torch.log(torch.tensor(t0)))
        self._inited.fill_(1)

    def forward(self, x: torch.Tensor):
        if self.bits >= 32:
            return x

        if self._inited.item() == 0:
            self._init_from(x)

        t = torch.exp(self.log_t).clamp_min(self.eps)
        s = (t / float(self.qp)).clamp_min(self.eps)

        x_c = x.clamp_min(0.0)
        if self.signed:
            x_c = torch.max(x, -t)
        x_c = torch.min(x_c, t)

        x_int = RoundSTE.apply(x_c / s)
        x_int = torch.clamp(x_int, self.qn, self.qp)
        x_q = x_int * s
        return x_q


class QuantConv2dTQT(nn.Module):
    def __init__(self, conv: nn.Conv2d, w_bits=8, a_bits=8, act_signed=False):
        super().__init__()

        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None

        self.wq = TQTQuantizer(bits=w_bits, signed=True)
        self.aq = TQTQuantizer(bits=a_bits, signed=act_signed)

    def forward(self, x):
        xq = self.aq(x)
        wq = self.wq(self.weight)
        return F.conv2d(xq, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)

@torch.no_grad()
def copy_espcn_fp32_to_tqt(fp32_model, tqt_model):
    def copy_conv(src: nn.Conv2d, dst: nn.Conv2d, name: str):
        dst.weight.copy_(src.weight)
        if src.bias is not None:
            dst.bias.copy_(src.bias)

    copy_conv(fp32_model.conv1, tqt_model.conv1, "conv1")
    copy_conv(fp32_model.conv2, tqt_model.conv2, "conv2")
    copy_conv(fp32_model.conv3, tqt_model.conv3, "conv3")


class NetTQT(nn.Module):
    def __init__(self, upscale_factor, w_bits=8, a_bits=8, in_channels=1, quantize_last=True, pretrained_fp32 = None):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2, bias=True)
        conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)
        conv3 = nn.Conv2d(32, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1, bias=True)

        self.conv1 = QuantConv2dTQT(conv1, w_bits=w_bits, a_bits=a_bits, act_signed=False)
        self.conv2 = QuantConv2dTQT(conv2, w_bits=w_bits, a_bits=a_bits, act_signed=False)
        self.conv3 = QuantConv2dTQT(conv3, w_bits=w_bits, a_bits=a_bits, act_signed=False)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        if pretrained_fp32 is not None:
            copy_espcn_fp32_to_tqt(pretrained_fp32, self)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x
