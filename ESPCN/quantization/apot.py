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


def _grad_scale(x, scale: float):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def build_apot_levels(bits: int, k: int = 2, signed: bool = True, device=None, dtype=None) -> torch.Tensor:
    if signed:
        mag_codes = (2 ** (bits - 1))
    else:
        mag_codes = (2 ** bits)

    m = int(round(mag_codes ** (1.0 / k)))
    m = max(2, m)
    m_list = [m] * k
    prod = 1
    for mi in m_list:
        prod *= mi
    while prod < mag_codes:
        m_list[-1] += 1
        prod = 1
        for mi in m_list:
            prod *= mi

    terms = []
    for mi in m_list:
        exps = torch.arange(0, mi, device=device, dtype=dtype)
        terms.append(2.0 ** (-exps))

    grids = torch.meshgrid(*terms, indexing="ij") if hasattr(torch, "meshgrid") else torch.meshgrid(*terms)
    sums = torch.zeros_like(grids[0])
    for g in grids:
        sums = sums + g
    levels = sums.reshape(-1)

    levels = torch.cat([torch.zeros(1, device=device, dtype=dtype), levels])
    levels = torch.unique(levels)
    levels, _ = torch.sort(levels)

    levels = levels / levels.max().clamp_min(1e-12)

    if signed:
        neg = -levels[1:]  # exclude -0
        all_levels = torch.cat([neg, levels])
        all_levels, _ = torch.sort(all_levels)
        return all_levels
    else:
        return levels

def nearest_level_bucketize(x_n: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    x_flat = x_n.reshape(-1)

    pos = torch.bucketize(x_flat, levels)

    pos0 = (pos - 1).clamp(0, levels.numel() - 1)
    pos1 = pos.clamp(0, levels.numel() - 1)

    l0 = levels[pos0]
    l1 = levels[pos1]

    choose1 = (x_flat - l0).abs() > (x_flat - l1).abs()
    out = torch.where(choose1, l1, l0)
    return out.view_as(x_n)

class APoTQuantizer(nn.Module):
    def __init__(
        self,
        bits: int = 8,
        k: int = 2,
        signed: bool = True,
        learn_scale: bool = True,
        init_scale: float | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.bits = bits
        self.k = k
        self.signed = signed
        self.eps = eps

        self.register_buffer("_inited", torch.tensor(0, dtype=torch.uint8))
        self._init_scale = init_scale

        s = 1.0 if init_scale is None else float(init_scale)
        if learn_scale:
            self.alpha = nn.Parameter(torch.tensor(s))
        else:
            self.register_buffer("alpha", torch.tensor(s))

        self.register_buffer("levels", torch.empty(0))

    @torch.no_grad()
    def _init_from(self, x: torch.Tensor):
        if self._init_scale is not None:
            if isinstance(self.alpha, nn.Parameter):
                self.alpha.fill_(float(self._init_scale))
            else:
                self.alpha.copy_(torch.tensor(float(self._init_scale), device=x.device, dtype=x.dtype))
        else:
            a0 = x.detach().abs().max() if self.signed else x.detach().max()
            a0 = a0.clamp_min(self.eps)
            if isinstance(self.alpha, nn.Parameter):
                self.alpha.data = a0.to(self.alpha.device, self.alpha.dtype)
            else:
                self.alpha.copy_(a0)

        self.levels = build_apot_levels(self.bits, k=self.k, signed=self.signed, device=x.device, dtype=x.dtype)
        self._inited.fill_(1)

    def forward(self, x: torch.Tensor):
        if self.bits >= 32:
            return x

        if self._inited.item() == 0 or self.levels.numel() == 0 or self.levels.device != x.device or self.levels.dtype != x.dtype:
            self._init_from(x)

        g = 1.0 / math.sqrt(x.numel())
        alpha = _grad_scale(self.alpha, g).clamp_min(self.eps)

        if self.signed:
            x_c = torch.clamp(x, -alpha, alpha)
            x_n = x_c / alpha
        
        else:
            x_c = x.clamp_min(0.0)
            x_c = torch.min(x_c, alpha)
            x_n = x_c / alpha

        x_q_levels = nearest_level_bucketize(x_n, self.levels)

        x_q = x_n + (x_q_levels - x_n).detach()
        return x_q * alpha


class QuantActAPoT(nn.Module):
    def __init__(self, bits=8, k=2, init_alpha=None):
        super().__init__()
        self.q = APoTQuantizer(bits=bits, k=k, signed=False, learn_scale=True, init_scale=init_alpha)

    def forward(self, x):
        return self.q(x)

@torch.no_grad()
def load_fp32_to_apot(fp32_model, apot_model):
    apot_model.conv1.weight.copy_(fp32_model.conv1.weight)
    if fp32_model.conv1.bias is not None:
        apot_model.conv1.bias.copy_(fp32_model.conv1.bias)

    apot_model.conv2.weight.copy_(fp32_model.conv2.weight)
    if fp32_model.conv2.bias is not None:
        apot_model.conv2.bias.copy_(fp32_model.conv2.bias)

    apot_model.conv3.weight.copy_(fp32_model.conv3.weight)
    if fp32_model.conv3.bias is not None:
        apot_model.conv3.bias.copy_(fp32_model.conv3.bias)

class NetAPoTActivations(nn.Module):
    def __init__(self, upscale_factor, a_bits=8, k=2, in_channels=1, quantize_input=False, pretrained_fp32=None):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self.q_in = APoTQuantizer(bits=a_bits, k=k, signed=False) if quantize_input else None
        self.q1 = QuantActAPoT(bits=a_bits, k=k)
        self.q2 = QuantActAPoT(bits=a_bits, k=k)
        
        if pretrained_fp32 is not None:
            load_fp32_to_apot(pretrained_fp32, self)

    def forward(self, x):
        if self.q_in is not None:
            x = self.q_in(x)

        x = self.relu(self.conv1(x))
        x = self.q1(x)

        x = self.relu(self.conv2(x))
        x = self.q2(x)

        x = self.pixel_shuffle(self.conv3(x))
        return x
