import argparse
import os
import time

import numpy as np
import torch

from model import SASRec
from utils import data_partition, evaluate
from benchmark import load_args, build_model, load_state, cpu_threads, measure_speed

class _Args:
    pass

def args_from_meta(meta, device='cpu'):
    a = _Args()
    a.device = device
    a.hidden_units = meta['hidden_units']
    a.num_blocks = meta['num_blocks']
    a.num_heads = meta['num_heads']
    a.maxlen = meta['maxlen']
    a.dropout_rate = 0.0
    a.norm_first = False
    a.quant_method = 'none'
    a.num_bits = meta.get('num_bits', 8)
    a.quant_full = False
    return a

def load_int8_into_fp32(int8_path, dataset_name=None, device='cpu'):
    obj = torch.load(int8_path, map_location='cpu', weights_only=False)
    if obj.get('format') != 'sasrec_int8_v1':
        raise ValueError(f'Неизвестный формат: {obj.get("format")}')
    meta = obj['meta']
    args = args_from_meta(meta, device=device)
    args.usernum = meta['usernum']
    args.itemnum = meta['itemnum']

    model = SASRec(meta['usernum'], meta['itemnum'], args).to(device)

    state_dict_fp32 = obj['fp32']

    state_dict_fp32 = {k: v for k, v in state_dict_fp32.items()
                       if 'quantizer' not in k or k.endswith('.weight')}
    missing, unexpected = model.load_state_dict(state_dict_fp32, strict=False)

    int8_w = obj['int8']
    scales = obj['scales']
    sd = model.state_dict()
    for name, w_int in int8_w.items():
        s = scales[name]
        w_fp = w_int.to(torch.float32) * s
        if name in sd:
            sd[name].copy_(w_fp.to(sd[name].dtype))
        else:
            print(f'  WARN: не нашёл {name} в model.state_dict()')

    return model, args

def disk_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2) if path and os.path.exists(path) else None

def fmt(v, fmtstr='{:.4f}'):
    return fmtstr.format(v) if v is not None else '-'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--int8', required=True, help='int8 pack файл (model.int8.pth)')
    parser.add_argument('--fp32', default=None, help='FP32 чекпоинт (опционально для сравнения)')
    parser.add_argument('--qat', default=None, help='QAT fake-quant чекпоинт (опционально)')
    parser.add_argument('--dataset', default='ml-1m')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', default=1, type=int)
    parser.add_argument('--num_runs', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    cli = parser.parse_args()

    print(f'Загружаю датасет {cli.dataset}...')
    dataset = data_partition(cli.dataset)
    _, _, _, usernum, itemnum = dataset
    print(f'  users={usernum}, items={itemnum}\n')

    rows = []

    def bench(label, model, args, src_path):
        model.eval()
        size = disk_size_mb(src_path)
        with cpu_threads(cli.threads):
            ms = measure_speed(model, args,
                               batch_size=cli.batch_size,
                               num_runs=cli.num_runs)
        print(f'  считаю NDCG@10 / HR@10 ...', end='', flush=True)
        ndcg, hr = evaluate(model, dataset, args)
        print(f'  NDCG@10={ndcg:.4f}  HR@10={hr:.4f}  '
              f'ms/batch={ms:.2f}  disk={fmt(size, "{:.3f}")} MB')
        rows.append({'label': label, 'disk_mb': size, 'ms': ms,
                     'ndcg10': ndcg, 'hr10': hr})

    if cli.fp32:
        print('=== FP32 baseline ===')
        train_dir = os.path.dirname(cli.fp32)
        args = load_args(train_dir, override_device=cli.device)
        args.usernum, args.itemnum = usernum, itemnum
        model = build_model(args, dataset)
        load_state(model, cli.fp32)
        bench('FP32 baseline', model, args, cli.fp32)
        del model

    if cli.qat:
        print('\n=== QAT fake-quant (LSQ) ===')
        train_dir = os.path.dirname(cli.qat)
        args = load_args(train_dir, override_device=cli.device)
        args.usernum, args.itemnum = usernum, itemnum
        model = build_model(args, dataset)
        load_state(model, cli.qat)
        bench('QAT fake-quant', model, args, cli.qat)
        del model

    print('\n=== Real INT8 (deployed) ===')
    model, args = load_int8_into_fp32(cli.int8, cli.dataset, device=cli.device)
    args.usernum, args.itemnum = usernum, itemnum
    bench('Real INT8', model, args, cli.int8)
    del model

    print('\n========== СВОДКА ==========')
    header = f'{"label":20s} {"disk MB":>10s} {"x compr":>8s} {"ms/batch":>10s} {"NDCG@10":>9s} {"HR@10":>9s}'
    print(header)
    print('-' * len(header))
    base_size = next((r['disk_mb'] for r in rows if r['label'] == 'FP32 baseline'), None)
    for r in rows:
        ratio = (base_size / r['disk_mb']) if (base_size and r['disk_mb']) else 1.0
        print(f'{r["label"]:20s} {fmt(r["disk_mb"], "{:>10.3f}"):>10s} '
              f'{ratio:>8.2f} {r["ms"]:>10.2f} '
              f'{r["ndcg10"]:>9.4f} {r["hr10"]:>9.4f}')

if __name__ == '__main__':
    main()
