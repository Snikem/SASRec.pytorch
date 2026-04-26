import argparse
import glob
import os
import re
import time
from contextlib import contextmanager

import numpy as np
import torch

from model import SASRec, PointWiseFeedForward
from utils import data_partition, evaluate

def str2bool(s):
    return str(s).lower() in {'1', 'true', 't', 'yes'}

class ArgsBag:
    pass

def load_args(train_dir, override_device='cpu'):
    args = ArgsBag()
    args_path = os.path.join(train_dir, 'args.txt')
    with open(args_path) as f:
        for line in f:
            k, v = line.strip().split(',', 1)
            setattr(args, k, _autoparse(v))

    if not hasattr(args, 'quant_method'):
        args.quant_method = 'pact' if getattr(args, 'use_pact', False) else 'none'
    if not hasattr(args, 'norm_first'):
        args.norm_first = False
    if not hasattr(args, 'quant_full'):
        args.quant_full = False
    args.device = override_device
    args.dropout_rate = 0.0
    return args

def _autoparse(v):
    if v in ('True', 'False'):
        return v == 'True'
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v

def find_checkpoint(train_dir):
    paths = glob.glob(os.path.join(train_dir, 'SASRec.epoch=*.pth'))
    if not paths:
        return None

    def epoch_of(p):
        m = re.search(r'epoch=(\d+)', os.path.basename(p))
        return int(m.group(1)) if m else 0
    return sorted(paths, key=epoch_of)[-1]

def build_model(args, dataset):
    user_train, _, _, usernum, itemnum = dataset
    args.usernum = usernum
    args.itemnum = itemnum
    model = SASRec(usernum, itemnum, args).to(args.device)

    if getattr(args, 'quant_method', 'none') == 'adaround':
        model.init_adaround_quantizers()
        model = model.to(args.device)
    return model

def load_state(model, ckpt):
    state = torch.load(ckpt, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  [WARN] missing keys: {len(missing)} (e.g. {missing[:2]})')
    if unexpected:
        print(f'  [WARN] unexpected keys: {len(unexpected)} (e.g. {unexpected[:2]})')

def _collect_param_kinds(model):
    kinds = {}
    names = {}
    quant_target_ids = (model.quant_target_param_ids()
                       if hasattr(model, 'quant_target_param_ids') else set())
    training_only_ids = (model.training_only_param_ids()
                        if hasattr(model, 'training_only_param_ids') else set())

    for n, p in model.named_parameters():
        if id(p) in quant_target_ids:
            kinds[id(p)] = 'quant_target'
        elif id(p) in training_only_ids:
            kinds[id(p)] = 'training_only'
        else:
            kinds[id(p)] = 'keep_fp32'
        names[id(p)] = n
    return kinds, names

def effective_size_bytes(model):
    kinds, _ = _collect_param_kinds(model)
    size_fp32_full = 0
    size_fp32_deploy = 0
    size_int8_deploy = 0
    n_quant_tensors = 0
    for p in model.parameters():
        n = p.numel()
        kind = kinds[id(p)]
        size_fp32_full += n * 4
        if kind == 'training_only':
            continue
        size_fp32_deploy += n * 4
        if kind == 'quant_target':
            size_int8_deploy += n * 1
            n_quant_tensors += 1
        else:
            size_int8_deploy += n * 4
    return size_fp32_full, size_fp32_deploy, size_int8_deploy, n_quant_tensors

def count_params(model):
    return sum(p.numel() for p in model.parameters())

@contextmanager
def cpu_threads(n):
    old = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(old)

def measure_speed(model, args, batch_size=128, num_runs=50, warmup=5):
    model.eval()
    rng = np.random.default_rng(0)
    users = rng.integers(1, args.usernum, size=(batch_size,))
    log_seqs = rng.integers(0, args.itemnum, size=(batch_size, args.maxlen))
    item_indices = rng.integers(1, args.itemnum, size=(batch_size, 100))

    with torch.no_grad():
        for _ in range(warmup):
            _ = model.predict(users, log_seqs, item_indices)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model.predict(users, log_seqs, item_indices)
    elapsed = time.perf_counter() - start
    return (elapsed / num_runs) * 1000.0

def auto_runs(dataset_name):
    candidates = sorted(glob.glob(f'{dataset_name}_*'))
    return [d for d in candidates if find_checkpoint(d) is not None]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m')
    parser.add_argument('--runs', nargs='*', default=None,
                        help='Список папок с обученными моделями. Если пусто — авто-поиск.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_runs', default=50, type=int)
    parser.add_argument('--threads', default=1, type=int, help='torch.set_num_threads')
    parser.add_argument('--eval', default='false', type=str2bool,
                        help='Оценивать NDCG@10/HR@10 (медленнее)')
    cli = parser.parse_args()

    runs = cli.runs or auto_runs(cli.dataset)
    if not runs:
        print('Не нашёл ни одной папки с обученными моделями.')
        print('Проверь, что в python/ лежит ml-1m_default/, ml-1m_pact_finetune/ и т.д.')
        return

    print(f'Бенчмарк на {cli.device} (threads={cli.threads})')
    print(f'Нашёл {len(runs)} run(ов): {runs}\n')

    print(f'Загружаю датасет {cli.dataset}...')
    dataset = data_partition(cli.dataset)
    _, _, _, usernum, itemnum = dataset
    print(f'  users={usernum}, items={itemnum}\n')

    results = []
    for run in runs:
        print(f'=== {run} ===')
        try:
            args = load_args(run, override_device=cli.device)
        except FileNotFoundError:
            print(f'  пропускаю: нет args.txt')
            continue

        ckpt = find_checkpoint(run)
        print(f'  checkpoint: {os.path.basename(ckpt)}')
        print(f'  quant_method: {getattr(args, "quant_method", "none")}')

        model = build_model(args, dataset)
        load_state(model, ckpt)
        model.eval()

        n_params = count_params(model)
        size_full, size_deploy_fp32, size_deploy_int8, n_quant_tensors = effective_size_bytes(model)
        size_full_mb = size_full / (1024 ** 2)
        size_dep_fp32_mb = size_deploy_fp32 / (1024 ** 2)
        size_dep_int8_mb = size_deploy_int8 / (1024 ** 2)
        compression = size_deploy_fp32 / size_deploy_int8 if size_deploy_int8 > 0 else 1.0
        print(f'  parameters: {n_params:,}')
        print(f'  size .pth (вкл. training-only): {size_full_mb:.3f} MB')
        print(f'  deploy FP32:                    {size_dep_fp32_mb:.3f} MB')
        print(f'  deploy INT8 (квантуемые → 1B): {size_dep_int8_mb:.3f} MB  '
              f'(сжатие x{compression:.3f}, квантуемых тензоров: {n_quant_tensors})')

        with cpu_threads(cli.threads):
            ms = measure_speed(model, args,
                               batch_size=cli.batch_size,
                               num_runs=cli.num_runs)
        print(f'  inference (CPU, fake-quant): {ms:.2f} ms/batch')

        ndcg, hr = (None, None)
        if cli.eval:
            print('  считаю NDCG@10 / HR@10 ...', end='', flush=True)
            ndcg, hr = evaluate(model, dataset, args)
            print(f'  NDCG@10={ndcg:.4f}  HR@10={hr:.4f}')

        results.append({
            'run': run,
            'method': getattr(args, 'quant_method', 'none'),
            'params': n_params,
            'size_full_mb': size_full_mb,
            'size_dep_fp32_mb': size_dep_fp32_mb,
            'size_dep_int8_mb': size_dep_int8_mb,
            'compression': compression,
            'ms_per_batch': ms,
            'ndcg10': ndcg,
            'hr10': hr,
        })
        print()

    print('\n=========== СВОДКА ===========')
    header = (f'{"run":40s} {"method":10s} {"params":>10s} '
              f'{"FP32(d)":>9s} {"INT8(d)":>9s} {"x":>6s} '
              f'{"ms/batch":>10s} {"NDCG@10":>9s} {"HR@10":>8s}')
    print(header)
    print('-' * len(header))
    for r in results:
        ndcg = f'{r["ndcg10"]:.4f}' if r["ndcg10"] is not None else '-'
        hr = f'{r["hr10"]:.4f}' if r["hr10"] is not None else '-'
        print(f'{r["run"]:40s} {r["method"]:10s} {r["params"]:>10,d} '
              f'{r["size_dep_fp32_mb"]:>9.3f} {r["size_dep_int8_mb"]:>9.3f} '
              f'{r["compression"]:>6.2f} {r["ms_per_batch"]:>10.2f} '
              f'{ndcg:>9s} {hr:>8s}')
    print('\nFP32(d) / INT8(d) — deploy-размер (без training-only параметров).')

    csv_path = 'benchmark_results.csv'
    with open(csv_path, 'w') as f:
        f.write('run,method,params,size_full_mb,size_deploy_fp32_mb,'
                'size_deploy_int8_mb,compression,ms_per_batch,ndcg10,hr10\n')
        for r in results:
            ndcg = f'{r["ndcg10"]:.6f}' if r["ndcg10"] is not None else ''
            hr = f'{r["hr10"]:.6f}' if r["hr10"] is not None else ''
            f.write(f'{r["run"]},{r["method"]},{r["params"]},'
                    f'{r["size_full_mb"]:.6f},{r["size_dep_fp32_mb"]:.6f},'
                    f'{r["size_dep_int8_mb"]:.6f},{r["compression"]:.6f},'
                    f'{r["ms_per_batch"]:.4f},{ndcg},{hr}\n')
    print(f'CSV сохранён: {csv_path}')

if __name__ == '__main__':
    main()
