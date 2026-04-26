import argparse
import os
import time
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data_utils import DatasetFromFolder
from model import Net

def load_state_dict_flexible(weights_path: str, device: torch.device) -> dict:
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def build_fp32_espcn(upscale_factor: int, device: torch.device) -> nn.Module:
    m = Net(upscale_factor=upscale_factor).to(device)
    m.eval()
    return m


def get_example_input(val_loader: DataLoader, device: torch.device, batch_size: int) -> torch.Tensor:
    lr, _ = next(iter(val_loader))
    x = lr[:batch_size].to(device)
    return x


def benchmark_cpu(model: nn.Module, x: torch.Tensor, iters: int = 300, warmup: int = 50, num_threads: int = 1) -> Tuple[float, float]:
    torch.set_num_threads(num_threads)
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    times.sort()
    mean_ms = sum(times) / len(times)
    median_ms = times[len(times) // 2]
    return mean_ms, median_ms


def file_size_mb(path: str) -> Optional[float]:
    if path is None or not os.path.exists(path):
        return None
    return os.path.getsize(path) / 1024 / 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["fp32", "int8"], help="what to benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="path to weights (.pth)")
    parser.add_argument("--upscale_factor", type=int, default=3)

    parser.add_argument("--val_dir", type=str, default="data/train")

    parser.add_argument("--batch_size", type=int, default=1, help="batch size for benchmarking input")

    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)

    parser.add_argument("--engine", type=str, default="fbgemm", choices=["fbgemm", "qnnpack"])

    parser.add_argument("--fp32_save", type=str, default=None, help="optional path to save fp32 state_dict for size compare")
    parser.add_argument("--int8_save", type=str, default=None, help="optional path to save int8 state_dict for size compare")

    opt = parser.parse_args()

    device = torch.device("cpu")

    val_set = DatasetFromFolder(
        opt.val_dir,
        upscale_factor=opt.upscale_factor,
        input_transform=transforms.ToTensor(),
        target_transform=transforms.ToTensor(),
    )
    val_loader = DataLoader(val_set, batch_size=max(opt.batch_size, 1), shuffle=False, num_workers=2)

    if opt.model_type == "fp32":
        model = build_fp32_espcn(opt.upscale_factor, device)
        state = load_state_dict_flexible(opt.model_path, device)
        model.load_state_dict(state, strict=True)

    else:
        torch.backends.quantized.engine = opt.engine

        model = None
        try:
            model = torch.jit.load(opt.model_path, map_location=device)
            model.eval()
        except Exception:
            model = torch.load(opt.model_path, map_location=device)
            if hasattr(model, "eval"):
                model.eval()

    x = get_example_input(val_loader, device, batch_size=opt.batch_size)

    mean_ms, median_ms = benchmark_cpu(model, x, iters=opt.iters, warmup=opt.warmup, num_threads=opt.threads)

    on_disk_mb = file_size_mb(opt.model_path)

    print(f"Model type: {opt.model_type}")
    print(f"Engine: {opt.engine if opt.model_type == 'int8' else 'n/a'}")
    print(f"Model file: {opt.model_path}")
    if on_disk_mb is not None:
        print(f"On-disk size: {on_disk_mb:.3f} MB")
    print(f"CPU threads: {opt.threads}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Latency: mean {mean_ms:.3f} ms | median {median_ms:.3f} ms  (iters={opt.iters}, warmup={opt.warmup})")

if __name__ == "__main__":
    main()
