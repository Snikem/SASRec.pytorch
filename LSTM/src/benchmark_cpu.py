import os
import time
import numpy as np
import torch

from sklearn.metrics import accuracy_score, roc_auc_score

def prepare_cpu_batches(loader, max_batches=None):
    """
    Заранее переносим батчи в список, чтобы benchmark не мерил DataLoader overhead.
    """
    batches = []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        input_ids = batch["input_ids"].cpu().long()

        attention_mask = None
        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"].cpu()

        labels = batch["label"].cpu().float()

        batches.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    return batches


@torch.inference_mode()
def evaluate_on_batches(model, batches):
    model.eval()
    model.cpu()

    all_logits = []
    all_labels = []

    for batch in batches:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = model(input_ids, attention_mask)

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    probs = torch.sigmoid(all_logits).numpy()
    y_true = all_labels.numpy()

    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, preds)
    roc_auc = roc_auc_score(y_true, probs)

    return acc, roc_auc


@torch.inference_mode()
def benchmark_cpu_model(
    model,
    batches,
    warmup=10,
    repeats=100,
):
    model.eval()
    model.cpu()

    # warmup
    for i in range(min(warmup, len(batches))):
        batch = batches[i % len(batches)]
        _ = model(batch["input_ids"], batch["attention_mask"])

    times = []
    total_samples = 0

    start_total = time.perf_counter()

    for i in range(repeats):
        batch = batches[i % len(batches)]

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        batch_size = input_ids.shape[0]

        start = time.perf_counter()
        _ = model(input_ids, attention_mask)
        end = time.perf_counter()

        times.append(end - start)
        total_samples += batch_size

    end_total = time.perf_counter()

    total_time = end_total - start_total

    times_np = np.array(times)

    result = {
        "mean_ms_per_batch": float(times_np.mean() * 1000),
        "median_ms_per_batch": float(np.median(times_np) * 1000),
        "p90_ms_per_batch": float(np.percentile(times_np, 90) * 1000),
        "p95_ms_per_batch": float(np.percentile(times_np, 95) * 1000),
        "samples_per_second": float(total_samples / total_time),
        "total_time_sec": float(total_time),
        "num_repeats": int(repeats),
        "batch_size": int(batches[0]["input_ids"].shape[0]),
    }

    return result


def disk_size_mb(path):
    if path is None or not os.path.exists(path):
        return None

    return os.path.getsize(path) / 1024 / 1024


def fmt(v, fmtstr="{:.4f}"):
    return fmtstr.format(v) if v is not None else "-"


def benchmark_three_versions(
    fp32_model,
    lsq_fake_model,
    int8_deployed_model,
    batches,
    fp32_path=None,
    lsq_path=None,
    int8_path=None,
    warmup=10,
    repeats=100,
):
    rows = []

    def run_one(label, model, path):
        model.cpu()
        model.eval()

        acc, roc_auc = evaluate_on_batches(model, batches)

        speed = benchmark_cpu_model(
            model,
            batches,
            warmup=warmup,
            repeats=repeats,
        )

        size = disk_size_mb(path)

        row = {
            "label": label,
            "disk_mb": size,
            "acc": acc,
            "roc_auc": roc_auc,
            "median_ms": speed["median_ms_per_batch"],
            "mean_ms": speed["mean_ms_per_batch"],
            "samples_per_second": speed["samples_per_second"],
        }

        rows.append(row)

        print(
            f"{label:22s} "
            f"acc={acc:.4f} "
            f"roc_auc={roc_auc if not np.isnan(roc_auc) else float('nan'):.4f} "
            f"median_ms={speed['median_ms_per_batch']:.2f} "
            f"disk={fmt(size, '{:.3f}')} MB"
        )

    print("=== FP32 baseline ===")
    run_one("FP32 baseline", fp32_model, fp32_path)

    print()
    print("=== QAT fake-quant LSQ ===")
    run_one("QAT fake-quant", lsq_fake_model, lsq_path)

    print()
    print("=== INT8 deployed pack ===")
    run_one("INT8 deployed", int8_deployed_model, int8_path)

    print()
    print("========== СВОДКА ==========")

    header = (
        f'{"label":22s} {"disk MB":>10s} {"x compr":>8s} '
        f'{"median ms":>10s} {"samples/s":>12s} {"acc":>8s} {"roc_auc":>8s}'
    )

    print(header)
    print("-" * len(header))

    base_size = next(
        (r["disk_mb"] for r in rows if r["label"] == "FP32 baseline"),
        None,
    )

    for r in rows:
        if base_size is not None and r["disk_mb"] is not None:
            ratio = base_size / r["disk_mb"]
        else:
            ratio = 1.0

        print(
            f'{r["label"]:22s} '
            f'{fmt(r["disk_mb"], "{:>10.3f}"):>10s} '
            f'{ratio:>8.2f} '
            f'{r["median_ms"]:>10.2f} '
            f'{r["samples_per_second"]:>12.2f} '
            f'{r["acc"]:>8.4f} '
            f'{r["roc_auc"]:>8.4f}'
        )

    return rows
