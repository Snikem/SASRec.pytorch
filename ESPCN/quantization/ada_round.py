import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _clamp(x, mn, mx):
    return torch.clamp(x, mn, mx)


class AdaRoundWeightQuantizer(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bits: int = 8,
        gamma: float = -0.1,
        zeta: float = 1.1,
        beta: float = 2.0 / 3.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.bits = bits
        self.gamma = gamma
        self.zeta = zeta
        self.beta = beta
        self.eps = eps

        self.qn = -(2 ** (bits - 1))
        self.qp = (2 ** (bits - 1)) - 1

        with torch.no_grad():
            maxv = weight.detach().abs().max().clamp_min(eps)
            s = maxv / float(self.qp)
        self.register_buffer("s", s)

        w_div = (weight.detach() / self.s)
        frac = w_div - torch.floor(w_div)
        frac = frac.clamp(0.0, 1.0)

        inv = ((frac - gamma) / (zeta - gamma)).clamp(1e-6, 1 - 1e-6)
        v0 = torch.log(inv / (1 - inv))
        self.v = nn.Parameter(v0)

        self.hard = False

    def set_hard(self, hard: bool = True):
        self.hard = hard

    def h(self):
        if self.hard:
            return (self.v > 0).to(self.v.dtype)
        return _clamp(torch.sigmoid(self.v) * (self.zeta - self.gamma) + self.gamma, 0.0, 1.0)

    def forward(self, w: torch.Tensor):
        w_div = w / self.s
        w_floor = torch.floor(w_div)
        w_int = w_floor + self.h()
        w_int = _clamp(w_int, self.qn, self.qp)
        w_q = w_int * self.s
        return w_q

    def regularization_loss(self):
        h = self.h()
        reg = torch.sum(1.0 - torch.abs(2.0 * h - 1.0).pow(self.beta))
        return reg

class AdaRoundConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, bits: int = 8):
        super().__init__()
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None

        self.wq = AdaRoundWeightQuantizer(self.weight, bits=bits)

    def set_hard(self, hard=True):
        self.wq.set_hard(hard)

    def forward(self, x):
        w_q = self.wq(self.weight)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def reg_loss(self):
        return self.wq.regularization_loss()

class NetAdaRound(nn.Module):
    def __init__(self, fp32_espcn: nn.Module, bits=8):
        super().__init__()
        self.relu = fp32_espcn.relu
        self.pixel_shuffle = fp32_espcn.pixel_shuffle

        self.conv1 = AdaRoundConv2d(fp32_espcn.conv1, bits=bits)
        self.conv2 = AdaRoundConv2d(fp32_espcn.conv2, bits=bits)
        self.conv3 = AdaRoundConv2d(fp32_espcn.conv3, bits=bits)
        
    def set_hard(self, hard=True):
        for m in self.modules():
            if isinstance(m, AdaRoundConv2d):
                m.set_hard(hard)

    def reg_loss(self):
        reg = 0.0
        for m in self.modules():
            if isinstance(m, AdaRoundConv2d):
                reg = reg + m.reg_loss()
        return reg

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

def adaround_optimize(fp32_model, adaround_model, calib_loader, iters=2000, lr=1e-2, lam=1e-4, device="cuda"):
    fp32_model.eval().to(device)
    adaround_model.train().to(device)

    for n, p in adaround_model.named_parameters():
        p.requires_grad = ("wq.v" in n)  # только v

    opt = torch.optim.Adam([p for p in adaround_model.parameters() if p.requires_grad], lr=lr)

    it = iter(calib_loader)
    for step in range(iters):
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(calib_loader)
            x, _ = next(it)

        x = x.to(device)

        with torch.no_grad():
            y_ref = fp32_model(x)

        y_q = adaround_model(x)
        rec = F.mse_loss(y_q, y_ref)
        reg = adaround_model.reg_loss()

        loss = rec + lam * reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    adaround_model.set_hard(True)
    adaround_model.eval()
    return adaround_model
