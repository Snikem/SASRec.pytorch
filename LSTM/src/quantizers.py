import math
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseQuantizer(nn.Module):
    def forward(self, x):
        raise NotImplementedError

class IdentityQuantizer(BaseQuantizer):
    def forward(self, x):
        return x

def round_good(x):
    return (x.round() - x).detach() + x # Чтобы текли градиенты

def grad_scaler(x, scale):
    return (x - x * scale).detach() + x * scale

def inverse_softplus(x, eps: float = 1e-8):
    x_tensor = torch.tensor(x)
    x_tensor = torch.clamp(x_tensor - eps, min=eps)
    return torch.log(torch.expm1(x_tensor))

class LSQQuantizer(BaseQuantizer):
    def __init__(self, bits: int = 8, eps: float = 1e-8):
        super().__init__()

        self.bits = bits
        raw_scale_init = inverse_softplus(1.0, eps)
        self.raw_scale = nn.Parameter(raw_scale_init)
        self.initialized = False
        self.eps = eps

        self.qmin = -(2 ** (bits - 1)) + 1
        self.qmax = (2 ** (bits - 1)) - 1

    def get_scale(self):
        return F.softplus(self.raw_scale)
        
    # Подбирает нормальный scale для начала по данным
    def init_scale(self, x):
        with torch.no_grad():
            mean_abs = x.detach().abs().mean()
            if mean_abs <= self.eps:
                return False
                
            init_value = 2 * mean_abs / math.sqrt(self.qmax)
            init_value = init_value.clamp_min(self.eps)
            raw_init = inverse_softplus(init_value, self.eps).to(
                device=self.raw_scale.device,
                dtype=self.raw_scale.dtype
            )

            self.raw_scale.copy_(raw_init)
            
        self.initialized = True

        return True

    def forward(self, x):
        if not self.initialized:
            was_initialized = self.init_scale(x)

            if not was_initialized:
                return x

        grad_factor = 1.0 / math.sqrt(x.numel() * self.qmax)

        scale = self.get_scale()
        scale = grad_scaler(scale, grad_factor)

        
        x_scaled = x / scale
        x_clamped = torch.clamp(x_scaled, self.qmin, self.qmax)
        x_rounded = round_good(x_clamped)
        x_fake_quantized = x_rounded * scale

        return x_fake_quantized

class PACTQuantizer(BaseQuantizer):
    def __init__(self, bits: int = 8, alpha_init: float = 10.0, eps: float = 1e-8):
        super().__init__()

        self.bits = bits
        self.eps = eps
        self.raw_alpha = nn.Parameter(inverse_softplus(alpha_init))

        self.register_buffer("qmin", torch.tensor(0.0))
        self.qmax = (2 ** bits) - 1

    def get_alpha(self):
        return F.softplus(self.raw_alpha)

    def forward(self, x):
        alpha = self.get_alpha()

        x_clipped = torch.clamp(x, self.qmin, alpha)
        scale = alpha / self.qmax

        x_scaled = x_clipped / scale
        x_rounded = round_good(x_scaled)
        
        x_fake_quantized = x_rounded * scale

        return x_fake_quantized

class TQTQuantizer(BaseQuantizer):
    def __init__(self, bits: int = 8, alpha_init: float = 10.0, eps: float = 1e-8):
        super().__init__()

        self.bits = bits
        self.eps = eps
        self.raw_alpha = nn.Parameter(inverse_softplus(alpha_init))

        self.qmin = -(2 ** (bits - 1)) + 1
        self.qmax = 2 ** (bits - 1) - 1

    def get_alpha(self):
        return F.softplus(self.raw_alpha)

    def forward(self, x):
        alpha = self.get_alpha()
        
        scale = alpha / self.qmax
        #scale = scale.clamp_min(self.eps)
        
        x_clipped = torch.clamp(x, -alpha, alpha)
        x_scaled = x_clipped / scale
        x_rounded = round_good(x_scaled)
        x_clamped = torch.clamp(x_rounded, self.qmin, self.qmax)
        x_fake_quantized = x_clamped * scale

        return x_fake_quantized

class ADAQuantizer(BaseQuantizer):
    def __init__(
            self,
            weights: torch.tensor,
            bits: int = 8,
            eps: float = 1e-8,
            val: bool = False
        
    ):
        super().__init__()
    
        self.bits = bits
        self.eps = eps

        self.qmin = -(2 ** (bits - 1)) + 1
        self.qmax = 2 ** (bits - 1) - 1

        self.val = val # Это для жесткого округдения во врмея инференса

        with torch.no_grad():
            max_abs = weights.detach().abs().max()
            scale = max_abs / self.qmax
            scale = scale.clamp_min(self.eps)

            self.register_buffer("scale", scale)
            
            w_scaled = weights / scale
            w_floor = torch.floor(w_scaled)
            w_fraction = w_scaled - w_floor

            w_fraction = w_fraction.clamp(1e-6, 1 - 1e-6)
            alpha_init = torch.log(w_fraction / (1 - w_fraction)) # Считаем логиты для сигмоиды
        
        self.alpha = nn.Parameter(alpha_init) # Выдает примерно те же значения для дробных частей

    def regularization(self):
        h = torch.sigmoid(self.alpha)
        return torch.mean(1.0 - torch.abs(2.0 * h - 1.0))
        
    def forward(self, weights):
        w_scaled = weights / self.scale

        if not self.val:
            w_round = torch.sigmoid(self.alpha)
        else:
            w_round = (torch.sigmoid(self.alpha) >= 0.5).float()

        w_floor = torch.floor(w_scaled)
        w_int = w_floor + w_round
        w_int = w_int.clamp(self.qmin, self.qmax)

        w_fake_quantized = w_int * self.scale

        return w_fake_quantized

from itertools import combinations
def build_apot_levels(bits: int = 8, num_powers: int = 8, max_addends: int = 2):
    max_levels = 2 ** bits

    powers = [2 ** -i for i in range(num_powers)]
    positive_levels = {0.0}

    for addends_count in range(1, max_addends + 1):
        for combo in combinations(powers, addends_count):
            value = sum(combo)
            if value <= 1.0:
                positive_levels.add(value)
            
    signed_levels = set()
    for value in positive_levels:
        signed_levels.add(value)
        signed_levels.add(-value)

    levels = torch.tensor(sorted(signed_levels))

    if levels.numel() > max_levels:
        raise ValueError("Bad arguments, too many values for so small bits")

    return levels

def nearest_apot_levels(x_norm: torch.tensor, levels: torch.tensor):
    x_flat = x_norm.reshape(-1)
    right_indices = torch.bucketize(x_flat, levels).clamp(0, len(levels) - 1)
    left_indices = (right_indices - 1).clamp(0, len(levels) - 1)

    left_values = levels[left_indices]
    right_values = levels[right_indices]

    left_dist = (x_flat - left_values).abs()
    right_dist = (right_values - x_flat).abs()

    nearest = torch.where(right_dist < left_dist, right_values, left_values)

    return nearest.view_as(x_norm)
        
    
class APoTQuantizer(BaseQuantizer):
    def __init__(
        self,
        bits: int = 8,
        eps: float = 1e-8,
        max_addends: int = 3, #сколько степеней двоек складываем
        num_powers: int = 8,
        alpha_init: float = 1.0,
    ):
        super().__init__()
        
        self.bits = bits
        self.eps = eps
        self.max_addends = max_addends

        levels = build_apot_levels(
            bits, num_powers, self.max_addends
        )
        self.register_buffer("levels", levels)

        raw_alpha_init = inverse_softplus(alpha_init)
        self.raw_alpha = nn.Parameter(raw_alpha_init)

    def get_alpha(self):
        return F.softplus(self.raw_alpha)
        
    def forward(self, x):
        alpha = self.get_alpha()
        x_clipped = x.clamp(-alpha, alpha)
        x_norm = x_clipped / alpha
        x_nearest = nearest_apot_levels(x_norm, self.levels)
        x_rounded = x_norm + (x_nearest - x_norm).detach()

        return x_rounded * alpha
        
