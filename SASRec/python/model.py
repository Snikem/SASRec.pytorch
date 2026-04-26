import numpy as np
import torch
import torch.nn.functional as F
import math
from itertools import product

class LSQFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, Qn, Qp):
        ctx.save_for_backward(x, s)
        ctx.Qn = Qn
        ctx.Qp = Qp

        x_scaled = x / s
        x_clipped = torch.clamp(x_scaled, Qn, Qp)
        x_rounded = torch.round(x_clipped)
        x_quant = x_rounded * s

        return x_quant

    @staticmethod
    def backward(ctx, grad_output):
        x, s = ctx.saved_tensors
        Qn, Qp = ctx.Qn, ctx.Qp

        x_scaled = x / s

        mask_lt = (x_scaled < Qn).float()
        mask_gt = (x_scaled > Qp).float()
        mask_mid = 1.0 - mask_lt - mask_gt

        grad_x = grad_output * mask_mid

        grad_s_mid = torch.round(x_scaled) - x_scaled
        grad_s_lt = Qn * torch.ones_like(x)
        grad_s_gt = Qp * torch.ones_like(x)

        grad_s = grad_output * (grad_s_mid * mask_mid + grad_s_lt * mask_lt + grad_s_gt * mask_gt)
        grad_s = grad_s.sum().view(1)

        return grad_x, grad_s, None, None

class LSQ(torch.nn.Module):
    def __init__(self, num_bits=8, is_unsigned=True):
        super(LSQ, self).__init__()
        self.num_bits = num_bits

        if is_unsigned:
            self.Qn = 0
            self.Qp = 2**num_bits - 1
        else:
            self.Qn = -(2**(num_bits - 1))
            self.Qp = 2**(num_bits - 1) - 1

        self.s = torch.nn.Parameter(torch.ones(1))
        self.is_initialized = False

    def init_scale(self, x):
        mean_val = x.detach().abs().mean()

        init_val = 2 * mean_val / math.sqrt(self.Qp) if mean_val > 0 else torch.tensor([0.01], device=x.device)
        self.s.data.copy_(init_val.to(self.s.device))
        self.is_initialized = True

    def forward(self, x):
        if not self.is_initialized and self.training:
            self.init_scale(x)

        s_pos = torch.abs(self.s) + 1e-8
        return LSQFunction.apply(x, s_pos, self.Qn, self.Qp)

class PACTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, k):
        ctx.save_for_backward(x, alpha)

        y = torch.clamp(x, min = 0, max = alpha.item())
        scale = (2**k - 1) / alpha
        y_q = torch.round( y * scale) / scale
        return y_q

    @staticmethod
    def backward(ctx, dLdy_q):

        x, alpha, = ctx.saved_tensors

        lower_bound      = x < 0
        upper_bound      = x > alpha

        x_range = ~(lower_bound|upper_bound)
        grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)
        return dLdy_q * x_range.float(), grad_alpha, None

class PACT(torch.nn.Module):
    def __init__(self, k=4, alpha_init=10.0):
        super(PACT, self).__init__()
        self.k = k

        self.alpha = torch.nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x):
        return PACTFunction.apply(x, self.alpha, self.k)

class AdaRoundWeightQuantizer(torch.nn.Module):

    def __init__(self, initial_weight, num_bits=8):
        super().__init__()
        self.num_bits = num_bits
        self.Qn = -(2 ** (num_bits - 1))
        self.Qp = 2 ** (num_bits - 1) - 1
        self.zeta = 1.1
        self.gamma = -0.1

        with torch.no_grad():
            max_val = initial_weight.detach().abs().max().clamp(min=1e-8)
            init_scale = max_val / self.Qp

        self.scale = torch.nn.Parameter(init_scale.view(1).clone())

        with torch.no_grad():
            w_div_s = initial_weight / init_scale
            rest = w_div_s - torch.floor(w_div_s)
            sigma = (rest - self.gamma) / (self.zeta - self.gamma)
            sigma = sigma.clamp(min=1e-4, max=1 - 1e-4)
            alpha_init = torch.log(sigma / (1.0 - sigma))
        self.alpha = torch.nn.Parameter(alpha_init.clone())

    def rectified_sigmoid(self):
        return torch.clamp(
            torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma,
            0.0,
            1.0,
        )

    def quant_loss(self, beta=2.0):
        h = self.rectified_sigmoid()
        return torch.sum(1.0 - torch.abs(2.0 * h - 1.0) ** beta)

    def forward(self, weight):
        s = self.scale.abs() + 1e-8
        h = self.rectified_sigmoid()

        w_div_s = weight / s
        w_int = torch.floor(w_div_s.detach()) + h + (w_div_s - w_div_s.detach())
        w_int_clamped = torch.clamp(w_int, self.Qn, self.Qp)
        return s * w_int_clamped

class UniformActQuant(torch.nn.Module):

    def __init__(self, num_bits=8, momentum=0.1):
        super().__init__()
        self.num_bits = num_bits
        self.Qp = 2 ** num_bits - 1
        self.momentum = momentum
        self.register_buffer("running_max", torch.tensor(1e-3))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                cur_max = x.detach().abs().max().clamp(min=1e-3)
                if not bool(self.initialized.item()):
                    self.running_max.fill_(cur_max.item())
                    self.initialized.fill_(True)
                else:
                    self.running_max.mul_(1 - self.momentum).add_(self.momentum * cur_max)
        rm = self.running_max.clamp(min=1e-3)
        s = rm / self.Qp
        x_clipped = torch.clamp(x, min=0.0, max=rm.item())
        x_q = torch.round(x_clipped / s) * s

        return x_clipped + (x_q - x_clipped).detach()

def _build_apot_levels(num_bits=8, n=2):
    k = num_bits // n
    groups = []
    for i in range(n):
        vals = [0.0]
        for j in range(2 ** k - 1):
            vals.append(2 ** (-(i + j * n)))
        groups.append(vals)
    levels = set()
    for combo in product(*groups):
        levels.add(round(sum(combo), 14))
    levels = sorted(levels)
    max_v = max(levels) if levels else 1.0
    if max_v > 0:
        levels = [v / max_v for v in levels]
    return levels

class APoTQuantizer(torch.nn.Module):

    def __init__(self, num_bits=8, signed=True):
        super().__init__()
        self.num_bits = num_bits
        self.signed = signed
        levels = _build_apot_levels(num_bits=num_bits, n=2)
        if signed:
            neg = [-v for v in levels[1:]][::-1]
            levels = neg + levels
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32))
        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.register_buffer("initialized", torch.tensor(False))

    def init_alpha(self, x):
        with torch.no_grad():
            if self.signed:
                v = x.detach().abs().max()
            else:
                v = x.detach().max()
            self.alpha.data.fill_(float(v.clamp(min=1e-3).item()))
        self.initialized.fill_(True)

    def forward(self, x):
        if not bool(self.initialized.item()) and self.training:
            self.init_alpha(x)

        a = self.alpha.abs() + 1e-8
        x_norm = x / a
        if self.signed:
            x_norm = torch.clamp(x_norm, -1.0, 1.0)
        else:
            x_norm = torch.clamp(x_norm, 0.0, 1.0)

        levels = self.levels
        L = levels.size(0)
        x_det = x_norm.detach().contiguous()
        idx_right = torch.bucketize(x_det, levels)
        idx_right = idx_right.clamp(max=L - 1)
        idx_left = (idx_right - 1).clamp(min=0)
        right = levels[idx_right]
        left = levels[idx_left]
        choose_right = (x_det - left).abs() > (right - x_det).abs()
        snapped = torch.where(choose_right, right, left)

        snapped_ste = x_norm + (snapped - x_norm).detach()
        return a * snapped_ste

class DSQQuantizer(torch.nn.Module):
    def __init__(self, num_bits=8, signed=True, alpha_init=1.0, k_init=2.5):
        super().__init__()
        self.num_bits = num_bits
        self.signed = signed
        if signed:
            self.Qn = -(2 ** (num_bits - 1))
            self.Qp = 2 ** (num_bits - 1) - 1
        else:
            self.Qn = 0
            self.Qp = 2 ** num_bits - 1
        self.alpha = torch.nn.Parameter(torch.tensor(float(alpha_init)))

        self.k = torch.nn.Parameter(torch.tensor(float(k_init)))
        self.register_buffer("initialized", torch.tensor(False))

    def init_alpha(self, x):
        with torch.no_grad():
            v = x.detach().abs().max() if self.signed else x.detach().max()
            self.alpha.data.fill_(float(v.clamp(min=1e-3).item()))
        self.initialized.fill_(True)

    def forward(self, x):
        if not bool(self.initialized.item()) and self.training:
            self.init_alpha(x)

        a = self.alpha.abs() + 1e-8

        if self.signed:
            x_clip = torch.minimum(torch.maximum(x, -a), a)
            delta = 2.0 * a / (self.Qp - self.Qn)
        else:
            zero = torch.zeros_like(a)
            x_clip = torch.minimum(torch.maximum(x, zero), a)
            delta = a / self.Qp

        scaled = x_clip / delta
        m = torch.floor(scaled.detach())
        frac = scaled - m

        k_pos = self.k.abs() + 1e-3

        soft_step = torch.tanh(k_pos * (frac - 0.5)) / torch.tanh(k_pos * 0.5)
        q_soft = m + 0.5 * (1.0 + soft_step)

        q_hard = torch.round(scaled.detach())
        q_ste = q_soft + (q_hard - q_soft).detach()

        return q_ste * delta

def make_weight_quantizer(method, num_bits=8):
    if method == 'lsq':
        return LSQ(num_bits=num_bits, is_unsigned=False)
    if method == 'apot':
        return APoTQuantizer(num_bits=num_bits, signed=True)
    if method == 'dsq':
        return DSQQuantizer(num_bits=num_bits, signed=True)

    return None

def quantizes_weights(method):
    return method in {'lsq', 'apot', 'dsq', 'adaround'}

class PointWiseFeedForward(torch.nn.Module):
    QUANT_METHODS = {'none', 'pact', 'lsq', 'adaround', 'apot', 'dsq'}

    def __init__(self, hidden_units, dropout_rate, quant_method='none', num_bits=8):
        super(PointWiseFeedForward, self).__init__()
        assert quant_method in self.QUANT_METHODS, f'Unknown quant_method: {quant_method}'
        self.quant_method = quant_method
        self.num_bits = num_bits

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

        if quant_method == 'pact':
            self.activation = PACT(k=num_bits)
        elif quant_method == 'lsq':
            self.activation = LSQ(num_bits=num_bits, is_unsigned=True)
        elif quant_method == 'adaround':

            self.activation = UniformActQuant(num_bits=num_bits)
        elif quant_method == 'apot':
            self.activation = APoTQuantizer(num_bits=num_bits, signed=False)
        elif quant_method == 'dsq':
            self.activation = DSQQuantizer(num_bits=num_bits, signed=False)
        else:
            self.activation = torch.nn.ReLU()

        self.weight_quantizer1 = None
        self.weight_quantizer2 = None
        if quant_method == 'lsq':
            self.weight_quantizer1 = LSQ(num_bits=num_bits, is_unsigned=False)
            self.weight_quantizer2 = LSQ(num_bits=num_bits, is_unsigned=False)
        elif quant_method == 'apot':
            self.weight_quantizer1 = APoTQuantizer(num_bits=num_bits, signed=True)
            self.weight_quantizer2 = APoTQuantizer(num_bits=num_bits, signed=True)
        elif quant_method == 'dsq':
            self.weight_quantizer1 = DSQQuantizer(num_bits=num_bits, signed=True)
            self.weight_quantizer2 = DSQQuantizer(num_bits=num_bits, signed=True)

    def init_adaround(self):
        assert self.quant_method == 'adaround'
        self.weight_quantizer1 = AdaRoundWeightQuantizer(
            self.conv1.weight.data.clone(), num_bits=self.num_bits
        )
        self.weight_quantizer2 = AdaRoundWeightQuantizer(
            self.conv2.weight.data.clone(), num_bits=self.num_bits
        )

    def _quantize_weight(self, conv, quantizer):
        if quantizer is None:
            return conv.weight
        return quantizer(conv.weight)

    def forward(self, inputs):

        x = inputs.transpose(-1, -2)

        if self.weight_quantizer1 is not None:
            q_w1 = self._quantize_weight(self.conv1, self.weight_quantizer1)
            x = F.conv1d(x, q_w1, self.conv1.bias)
        else:
            x = self.conv1(x)

        x = self.dropout1(x)
        x = self.activation(x)

        if self.weight_quantizer2 is not None:
            q_w2 = self._quantize_weight(self.conv2, self.weight_quantizer2)
            x = F.conv1d(x, q_w2, self.conv2.bias)
        else:
            x = self.conv2(x)

        x = self.dropout2(x)
        return x.transpose(-1, -2)

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        quant_method = getattr(args, 'quant_method', 'none')
        num_bits = getattr(args, 'num_bits', 8)

        quant_full = bool(getattr(args, 'quant_full', False)) and quantizes_weights(quant_method)
        self.quant_method = quant_method
        self.quant_full = quant_full
        self._num_bits = num_bits

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(
                args.hidden_units,
                args.dropout_rate,
                quant_method=quant_method,
                num_bits=args.num_bits
            )
            self.forward_layers.append(new_fwd_layer)

        self.item_emb_quantizer = None
        self.pos_emb_quantizer = None
        self.attn_in_quantizers = torch.nn.ModuleList()
        self.attn_out_quantizers = torch.nn.ModuleList()
        if quant_full and quant_method != 'adaround':
            self.item_emb_quantizer = make_weight_quantizer(quant_method, num_bits)
            self.pos_emb_quantizer = make_weight_quantizer(quant_method, num_bits)
            for _ in range(args.num_blocks):
                self.attn_in_quantizers.append(make_weight_quantizer(quant_method, num_bits))
                self.attn_out_quantizers.append(make_weight_quantizer(quant_method, num_bits))

    def _embed(self, emb, quantizer, idx):
        w = quantizer(emb.weight) if quantizer is not None else emb.weight
        return F.embedding(idx, w, padding_idx=emb.padding_idx)

    def _attn(self, layer_idx, query, key, value, attn_mask):
        attn = self.attention_layers[layer_idx]
        in_q = (self.attn_in_quantizers[layer_idx]
                if layer_idx < len(self.attn_in_quantizers) else None)
        out_q = (self.attn_out_quantizers[layer_idx]
                 if layer_idx < len(self.attn_out_quantizers) else None)
        if in_q is None and out_q is None:
            out, _ = attn(query, key, value, attn_mask=attn_mask)
            return out

        iw = in_q(attn.in_proj_weight) if in_q is not None else attn.in_proj_weight
        ow = out_q(attn.out_proj.weight) if out_q is not None else attn.out_proj.weight
        out, _ = F.multi_head_attention_forward(
            query, key, value,
            embed_dim_to_check=attn.embed_dim,
            num_heads=attn.num_heads,
            in_proj_weight=iw,
            in_proj_bias=attn.in_proj_bias,
            bias_k=attn.bias_k,
            bias_v=attn.bias_v,
            add_zero_attn=attn.add_zero_attn,
            dropout_p=attn.dropout if self.training else 0.0,
            out_proj_weight=ow,
            out_proj_bias=attn.out_proj.bias,
            training=self.training,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return out

    def log2feats(self, log_seqs):
        idx_seq = torch.LongTensor(log_seqs).to(self.dev)
        seqs = self._embed(self.item_emb, self.item_emb_quantizer, idx_seq)
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])

        poss *= (log_seqs != 0)
        seqs = seqs + self._embed(self.pos_emb, self.pos_emb_quantizer,
                                  torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs = self._attn(i, x, x, x, attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs = self._attn(i, seqs, seqs, seqs, attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits

    def _iter_adaround_quantizers(self):
        for layer in self.forward_layers:
            if isinstance(layer, PointWiseFeedForward):
                for q in (layer.weight_quantizer1, layer.weight_quantizer2):
                    if isinstance(q, AdaRoundWeightQuantizer):
                        yield q
        if isinstance(self.item_emb_quantizer, AdaRoundWeightQuantizer):
            yield self.item_emb_quantizer
        if isinstance(self.pos_emb_quantizer, AdaRoundWeightQuantizer):
            yield self.pos_emb_quantizer
        for q in self.attn_in_quantizers:
            if isinstance(q, AdaRoundWeightQuantizer):
                yield q
        for q in self.attn_out_quantizers:
            if isinstance(q, AdaRoundWeightQuantizer):
                yield q

    def init_adaround_quantizers(self):

        for layer in self.forward_layers:
            if isinstance(layer, PointWiseFeedForward) and layer.quant_method == 'adaround':
                layer.init_adaround()

        if not self.quant_full or self.quant_method != 'adaround':
            return
        nb = self._num_bits
        self.item_emb_quantizer = AdaRoundWeightQuantizer(
            self.item_emb.weight.data.clone(), num_bits=nb)
        self.pos_emb_quantizer = AdaRoundWeightQuantizer(
            self.pos_emb.weight.data.clone(), num_bits=nb)

        self.attn_in_quantizers = torch.nn.ModuleList([
            AdaRoundWeightQuantizer(attn.in_proj_weight.data.clone(), num_bits=nb)
            for attn in self.attention_layers
        ])
        self.attn_out_quantizers = torch.nn.ModuleList([
            AdaRoundWeightQuantizer(attn.out_proj.weight.data.clone(), num_bits=nb)
            for attn in self.attention_layers
        ])

    def adaround_loss(self, beta=2.0):
        loss = torch.zeros(1, device=self.dev)
        n_total = 0
        for q in self._iter_adaround_quantizers():
            loss = loss + q.quant_loss(beta=beta)
            n_total += q.alpha.numel()
        if n_total > 0:
            loss = loss / n_total
        return loss

    def quant_target_param_ids(self):
        ids = set()
        for layer in self.forward_layers:
            if isinstance(layer, PointWiseFeedForward) and layer.quant_method != 'none':
                ids.add(id(layer.conv1.weight))
                ids.add(id(layer.conv2.weight))
        if self.quant_full:
            ids.add(id(self.item_emb.weight))
            ids.add(id(self.pos_emb.weight))
            for attn in self.attention_layers:
                ids.add(id(attn.in_proj_weight))
                ids.add(id(attn.out_proj.weight))
        return ids

    def training_only_param_ids(self):
        ids = set()
        for q in self._iter_adaround_quantizers():
            ids.add(id(q.alpha))
        return ids
