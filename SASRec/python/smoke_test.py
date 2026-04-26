import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from model import SASRec

class A:

    def __init__(self, method, quant_full=False):
        self.device = 'cpu'
        self.maxlen = 16
        self.hidden_units = 16
        self.num_blocks = 1
        self.num_heads = 2
        self.dropout_rate = 0.1
        self.norm_first = False
        self.quant_method = method
        self.num_bits = 8
        self.quant_full = quant_full

def run(method, quant_full=False):
    args = A(method, quant_full=quant_full)
    torch.manual_seed(0)
    np.random.seed(0)

    model = SASRec(user_num=10, item_num=20, args=args).to('cpu')
    for _, p in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(p.data)
        except Exception:
            pass

    if method == 'adaround':
        model.init_adaround_quantizers()

    model.train()
    bs, sl = 4, args.maxlen
    log_seqs = np.random.randint(1, 20, size=(bs, sl))
    pos = np.random.randint(1, 20, size=(bs, sl))
    neg = np.random.randint(1, 20, size=(bs, sl))
    users = np.arange(1, bs + 1)

    pl, nl = model(users, log_seqs, pos, neg)
    loss = (pl - 1.0).pow(2).mean() + (nl + 1.0).pow(2).mean()
    if method == 'adaround':
        loss = loss + 0.01 * model.adaround_loss().squeeze()

    loss.backward()
    has_nan = any(
        torch.isnan(p.grad).any().item()
        for p in model.parameters()
        if p.grad is not None
    )
    no_grad_count = sum(1 for p in model.parameters() if p.grad is None)
    qt = len(model.quant_target_param_ids())
    label = f'{method}{"+full" if quant_full else ""}'
    print(f'  {label:14s}  loss={loss.item():.4f}  '
          f'nan_in_grad={has_nan}  no_grad_params={no_grad_count}  '
          f'quant_targets={qt}')

    model.eval()
    with torch.no_grad():
        item_idx = np.random.randint(1, 20, size=(bs, 5))
        _ = model.predict(users, log_seqs, item_idx)

def main():
    print('Smoke test — forward+backward для каждого метода\n')
    for m in ['none', 'pact', 'lsq', 'adaround', 'apot', 'dsq']:
        try:
            run(m)
        except Exception as e:
            print(f'  {m:10s}  FAILED: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
    print('\nЕсли везде nan_in_grad=False и нет FAILED — код в порядке.')

if __name__ == '__main__':
    main()
