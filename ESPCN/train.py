import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from math import log10
import matplotlib.pyplot as plt

from data_utils import DatasetFromFolder
from model import Net
from quantization.lsq import NetLSQ
from quantization.pact import NetPACTActivations
from quantization.tqt import NetTQT
from quantization.apot import NetAPoTActivations

def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * log10((max_val * max_val) / mse)


def build_model(kind: str, upscale_factor: int, pretrained_model = None, w_bits=8, a_bits=8, quantize_last=False):
    if kind == "fp32":
        return Net(upscale_factor=upscale_factor)
    if kind == "lsq":
        return NetLSQ(
            upscale_factor=upscale_factor,
            w_bits=w_bits, a_bits=a_bits,
            quantize_last=quantize_last,
            pretrained_fp32 = pretrained_model,
        )
    if kind == "pact":
        return NetPACTActivations(
            upscale_factor=upscale_factor,
            a_bits=a_bits,
            quantize_input=quantize_last,  # yes.
            pretrained_fp32 = pretrained_model,
        )
    if kind == "tqt":
        return NetTQT(
            upscale_factor=upscale_factor,
            w_bits=w_bits, a_bits=a_bits,
            quantize_last=quantize_last,
            pretrained_fp32=pretrained_model,
        )
    if kind == "apot":
        return NetAPoTActivations(
            upscale_factor=upscale_factor,
            a_bits=a_bits,
            quantize_input=quantize_last,
            pretrained_fp32=pretrained_model
        )
    raise ValueError(f"Unknown model kind: {kind}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Super Resolution')
    parser.add_argument('--upscale_factor', default=3, type=int, help='super resolution upscale factor')
    parser.add_argument('--num_epochs', default=100, type=int, help='super resolution epochs number')
    parser.add_argument('--plot_path', default='training_curves.png', type=str, help='where to save curves plot')
    parser.add_argument('--model', default='fp32', type=str, help='which model to train, possible: fp32, lsq, pact, ...')
    parser.add_argument('--from_pretrained', type=str, help='checkpoint of pretrained fp32 model')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--quantize_last', default=False, type=bool, help='whether to quantize the last conv2d layer')
    parser.add_argument('--w_bits', default=8, type=int, help='bits to quantize weights')
    parser.add_argument('--a_bits', default=8, type=int, help='bits to quantize activations')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    PLOT_PATH = opt.plot_path
    MODEL_TYPE = opt.model
    FROM_PRETRAINED = opt.from_pretrained
    LR = opt.lr
    QUANTIZE_LAST = opt.quantize_last
    W_BITS = opt.w_bits
    A_BITS = opt.a_bits

    train_set = DatasetFromFolder('data/train', upscale_factor=UPSCALE_FACTOR, input_transform=transforms.ToTensor(),
                                  target_transform=transforms.ToTensor())
    val_set = DatasetFromFolder('data/val', upscale_factor=UPSCALE_FACTOR, input_transform=transforms.ToTensor(),
                                target_transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=64, shuffle=False)

    pretrained_model = None
    if FROM_PRETRAINED is not None:
        ckpt = torch.load(FROM_PRETRAINED)
        pretrained_model = Net(3)
        pretrained_model.load_state_dict(ckpt)
    model = build_model(MODEL_TYPE, UPSCALE_FACTOR, pretrained_model, w_bits=W_BITS, a_bits=A_BITS, quantize_last=QUANTIZE_LAST)
    criterion = nn.MSELoss()
    device = "cpu"
    if torch.cuda.is_available():
        device="cuda:0"
        model = model.cuda()
        criterion = criterion.cuda()

    print('# parameters:', sum(param.numel() for param in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
    

    epochs_hist = []
    train_loss_hist = []
    val_loss_hist = []
    val_psnr_hist = []

    plt.ion()
    fig, (ax_loss, ax_psnr) = plt.subplots(1, 2, figsize=(12, 4))

    def redraw():
        ax_loss.clear()
        ax_psnr.clear()

        ax_loss.plot(epochs_hist, train_loss_hist, marker='o', label='train loss')
        ax_loss.plot(epochs_hist, val_loss_hist, marker='o', label='val loss')
        ax_loss.set_title("Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("MSE")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend()

        ax_psnr.plot(epochs_hist, val_psnr_hist, marker='o', label='val PSNR')
        ax_psnr.set_title("PSNR")
        ax_psnr.set_xlabel("Epoch")
        ax_psnr.set_ylabel("dB")
        ax_psnr.grid(True, alpha=0.3)
        ax_psnr.legend()

        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=150)
    
    best_val_psnr = -1.0
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]")
        for lr_img, hr_img in pbar:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            optimizer.zero_grad(set_to_none=True)
            sr_img = model(lr_img)
            loss = criterion(sr_img, hr_img)
            loss.backward()
            optimizer.step()

            bs = lr_img.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

            train_loss_avg = train_loss_sum / max(1, train_count)
            pbar.set_postfix(train_loss=f"{train_loss_avg:.6f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss_avg = train_loss_sum / max(1, train_count)
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [val]")
            for lr_img, hr_img in pbar:
                lr_img = lr_img.to(device)
                hr_img = hr_img.to(device)

                sr_img = model(lr_img)
                loss = criterion(sr_img, hr_img)

                bs = lr_img.size(0)
                val_loss_sum += loss.item() * bs
                val_count += bs

                val_loss_avg = val_loss_sum / max(1, val_count)
                val_psnr = psnr_from_mse(val_loss_avg, max_val=1.0)  # ToTensor -> [0,1]
                pbar.set_postfix(val_loss=f"{val_loss_avg:.6f}", psnr=f"{val_psnr:.3f}dB")

        val_loss_avg = val_loss_sum / max(1, val_count)
        val_psnr = psnr_from_mse(val_loss_avg, max_val=1.0)

        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS:03d} | "
            f"train_loss: {train_loss_avg:.6f} | "
            f"val_loss: {val_loss_avg:.6f} | "
            f"val_psnr: {val_psnr:.3f} dB | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e}"
        )
        
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), PLOT_PATH[:-4] + "_best_psnr.pth")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), PLOT_PATH[:-4] + "_best_loss.pth")
        
        epochs_hist.append(epoch)
        train_loss_hist.append(train_loss_avg)
        val_loss_hist.append(val_loss_avg)
        val_psnr_hist.append(val_psnr)
        redraw()

    torch.save(model.state_dict(), PLOT_PATH[:-4] + "_last.pth")
