import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearQuantized(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        quant_factory,
        bias: bool = True
    ):
        super().__init__()

        self.weights = nn.Parameter(torch.empty(out_size, in_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_size))
        else:
            self.bias = None

        self.reset_parameters()

        self.weight_quantizer = quant_factory.make_weights(self.weights, "linear_weights")
        self.input_quantizer = quant_factory.make_activations("linear_input")

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights)

        if self.bias is not None:
            out_size = self.weights.shape[1]
            stdv = 1. / math.sqrt(out_size)
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x):
        x_q = self.input_quantizer(x)
        w_q = self.weight_quantizer(self.weights)

        result = F.linear(x_q, w_q, self.bias)

        return result

class LSTMCellQuantized(nn.Module):
    def __init__(
        self,
        inp_size,
        hidden_size,
        quant_factory,
        quantized_cell_state: bool = False
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.quantized_cell_state = quantized_cell_state
        
        self.weights_inp = nn.Parameter(torch.empty(4 * hidden_size, inp_size))
        self.weights_hid = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        
        self.bias_hid = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_inp = nn.Parameter(torch.empty(4 * hidden_size))

        self.reset_parameters()

        self.x_quantizer = quant_factory.make_activations("lstm_x")
        self.h_quantizer = quant_factory.make_activations("lstm_h")
        self.c_quantizer = quant_factory.make_activations("lstm_c")

        self.weights_inp_quantizer = quant_factory.make_weights(self.weights_inp, "lstm_weights_inp")
        self.weights_hid_quantizer = quant_factory.make_weights(self.weights_hid, "lstm_weights_hid")

    def reset_parameters(self):
        stdv = 1. / (math.sqrt(self.hidden_size))

        for layer in [self.weights_inp, self.weights_hid, self.bias_hid, self.bias_inp]:
            nn.init.uniform_(layer, -stdv, stdv)

    def forward(self, x, state):
        x_q = self.x_quantizer(x)
        h, c = state
        h_q = self.h_quantizer(h)
        
        if self.quantized_cell_state:
            c_q = self.c_quantizer(c)
        else:
            c_q = c

        weights_inp_q = self.weights_inp_quantizer(self.weights_inp)
        weights_hid_q = self.weights_hid_quantizer(self.weights_hid)

        gates = F.linear(x_q, weights_inp_q, self.bias_inp) + F.linear(h_q, weights_hid_q, self.bias_hid)

        i_gate, f_gate, c_gate, o_gate = gates.chunk(4, dim=1)

        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        c_gate = torch.tanh(c_gate)
        o_gate = torch.sigmoid(o_gate)

        c_new = f_gate * c_q + i_gate * c_gate
        h_new = o_gate * torch.tanh(c_new)

        return h_new, c_new

class LSTMQuantized(nn.Module):
    def __init__(
        self,
        inp_size,
        hidden_size,
        quant_factory,
        quantized_cell_state: bool = False
    ):
        super().__init__()

        self.cell = LSTMCellQuantized(inp_size, hidden_size, quant_factory, quantized_cell_state)
        self.hidden_size = hidden_size

    def forward(self, x, state=None):
        batch_size, seq_len, _ = x.shape

        if state is None:
            h = x.new_zeros(batch_size, self.hidden_size)
            c = x.new_zeros(batch_size, self.hidden_size)
        else:
            h, c = state
        
        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :], (h, c))
            outputs.append(h.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)

        return outputs, (h, c)

class LSTMClassifierQuantized(nn.Module):
    def __init__(
        self,
        emb_size,
        #inp_size,
        hidden_size,
        quant_factory,
        pad_idx,
        vocab_size=30522,
        quantized_cell_state=True
    ):
        super().__init__()
        
        self.emb = nn.Embedding(vocab_size, emb_size, pad_idx)
        self.lstm = LSTMQuantized(emb_size, hidden_size, quant_factory, quantized_cell_state)
        self.linear = LinearQuantized(hidden_size, 1, quant_factory)

    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.long()
        emb = self.emb(input_ids)
        outputs, (h, c) = self.lstm(emb)

        if attention_mask is not None: # Потому что input_ids = [batch, max_seq_len, embeddings] куча паддингов
            lenghts = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
            h = outputs[batch_indices, lenghts]

        logits = self.linear(h).squeeze(-1)

        return logits
        
