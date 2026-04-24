import numpy as np
import torch
import torch.nn.functional as F
import math

class LSQFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, Qn, Qp):
        ctx.save_for_backward(x, s)
        ctx.Qn = Qn
        ctx.Qp = Qp
        
        # Масштабируем, клипаем, округляем и возвращаем масштаб
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
        
        # Маски для STE
        mask_lt = (x_scaled < Qn).float()
        mask_gt = (x_scaled > Qp).float()
        mask_mid = 1.0 - mask_lt - mask_gt
        
        # Градиент по входу (активациям/весам)
        grad_x = grad_output * mask_mid
        
        # Градиент по шагу квантования (scale)
        grad_s_mid = torch.round(x_scaled) - x_scaled
        grad_s_lt = Qn * torch.ones_like(x)
        grad_s_gt = Qp * torch.ones_like(x)
        
        grad_s = grad_output * (grad_s_mid * mask_mid + grad_s_lt * mask_lt + grad_s_gt * mask_gt)
        grad_s = grad_s.sum().view(1) # Скаляр
        
        return grad_x, grad_s, None, None

class LSQ(torch.nn.Module):
    def __init__(self, num_bits=8, is_unsigned=True):
        super(LSQ, self).__init__()
        self.num_bits = num_bits
        
        # Если is_unsigned=True, LSQ работает как ReLU (от 0 до 2^k - 1)
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
        # Инициализируем на том же устройстве (CPU/GPU), что и тензор x
        init_val = 2 * mean_val / math.sqrt(self.Qp) if mean_val > 0 else torch.tensor([0.01], device=x.device)
        self.s.data.copy_(init_val.to(self.s.device))
        self.is_initialized = True

    def forward(self, x):
        if not self.is_initialized and self.training:
            self.init_scale(x)
            
        s_pos = torch.abs(self.s) + 1e-8 # Защита от деления на ноль и отрицательного масштаба
        return LSQFunction.apply(x, s_pos, self.Qn, self.Qp)

class PACTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, k):
        ctx.save_for_backward(x, alpha)
        # y_1 = 0.5 * ( torch.abs(x).detach() - torch.abs(x - alpha).detach() + alpha.item() )
        y = torch.clamp(x, min = 0, max = alpha.item())
        scale = (2**k - 1) / alpha
        y_q = torch.round( y * scale) / scale
        return y_q

    @staticmethod
    def backward(ctx, dLdy_q):
        # Backward function, I borrowed code from
        # https://github.com/obilaniu/GradOverride/blob/master/functional.py
        # We get dL / dy_q as a gradient
        x, alpha, = ctx.saved_tensors
        # Weight gradient is only valid when [0, alpha]
        # Actual gradient for alpha,
        # By applying Chain Rule, we get dL / dy_q * dy_q / dy * dy / dalpha
        # dL / dy_q = argument,  dy_q / dy * dy / dalpha = 0, 1 with x value range 
        lower_bound      = x < 0
        upper_bound      = x > alpha
        # x_range       = 1.0-lower_bound-upper_bound
        x_range = ~(lower_bound|upper_bound)
        grad_alpha = torch.sum(dLdy_q * torch.ge(x, alpha).float()).view(-1)
        return dLdy_q * x_range.float(), grad_alpha, None

class PACT(torch.nn.Module):
    def __init__(self, k=4, alpha_init=10.0):
        super(PACT, self).__init__()
        self.k = k
        # alpha — это обучаемый параметр, поэтому оборачиваем его в nn.Parameter
        self.alpha = torch.nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x):
        return PACTFunction.apply(x, self.alpha, self.k)
    
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate, quant_method='none', num_bits=8):
        self.quant_method = quant_method
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)

        if quant_method == 'pact':
            self.activation = PACT(k=num_bits)
        elif quant_method == 'lsq':
            # is_unsigned=True обрезает все, что меньше 0 (заменяет ReLU)
            self.activation = LSQ(num_bits=num_bits, is_unsigned=True)
        else:
            self.activation = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

        if quant_method == 'lsq':
            # Для весов всегда используем is_unsigned=False (от -128 до 127)
            self.weight_quantizer1 = LSQ(num_bits=num_bits, is_unsigned=False)
            self.weight_quantizer2 = LSQ(num_bits=num_bits, is_unsigned=False)

    def forward(self, inputs):
       # Подготавливаем тензор для Conv1D (N, C, L)
        x = inputs.transpose(-1, -2)
        
        # --- Первый слой Conv1D ---
        if self.quant_method == 'lsq':
            # 1. Квантуем веса "на лету"
            q_weight1 = self.weight_quantizer1(self.conv1.weight)
            # 2. Используем функциональную свертку (сохраняем граф для backprop)
            x = F.conv1d(x, q_weight1, self.conv1.bias)
        else:
            x = self.conv1(x)
            
        x = self.dropout1(x)
        x = self.activation(x) # Здесь применяется PACT или LSQ для активаций
        
        # --- Второй слой Conv1D ---
        if self.quant_method == 'lsq':
            q_weight2 = self.weight_quantizer2(self.conv2.weight)
            x = F.conv1d(x, q_weight2, self.conv2.bias)
        else:
            x = self.conv2(x)
            
        x = self.dropout2(x)
        
        # Возвращаем размерность обратно
        outputs = x.transpose(-1, -2) 
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        quant_method = getattr(args, 'quant_method', 'none')

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

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x,
                                                attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                                attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
