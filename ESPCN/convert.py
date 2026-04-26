import argparse

import torch
import torch.nn as nn
import torch.ao.quantization as tq
import torchvision.transforms as transforms

from data_utils import DatasetFromFolder
from torch.utils.data import DataLoader

from model import Net
from quantization.lsq import NetLSQ

@torch.no_grad()
def export_lsq_weights_to_fp32_net(lsq: nn.Module, fp32: nn.Module):
    def get_weight_bias(layer):
        w = layer.weight
        b = layer.bias if getattr(layer, "bias", None) is not None else None
        return w, b

    w, b = get_weight_bias(lsq.conv1)
    fp32.conv1.weight.copy_(w)
    if fp32.conv1.bias is not None and b is not None:
        fp32.conv1.bias.copy_(b)

    w, b = get_weight_bias(lsq.conv2)
    fp32.conv2.weight.copy_(w)
    if fp32.conv2.bias is not None and b is not None:
        fp32.conv2.bias.copy_(b)

    w, b = get_weight_bias(lsq.conv3)
    fp32.conv3.weight.copy_(w)
    if fp32.conv3.bias is not None and b is not None:
        fp32.conv3.bias.copy_(b)

class QuantWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.quant = tq.QuantStub()
        self.m = m
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.m(x)
        x = self.dequant(x)
        return x


def to_int8_cpu_ptq(fp32_model: nn.Module, calib_loader, num_calib_batches=50, engine="fbgemm"):
    torch.backends.quantized.engine = engine
    m = QuantWrapper(fp32_model.cpu().eval())
    m.qconfig = tq.get_default_qconfig(engine)
    
    tq.prepare(m, inplace=True)

    with torch.no_grad():
        for i, (lr, hr) in enumerate(calib_loader):
            if i >= num_calib_batches:
                break
            m(lr.cpu())

    int8 = tq.convert(m.eval(), inplace=False)
    return int8



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Trained Quantized model to int8')
    parser.add_argument('--trained_model', type=str, help='LSQ model to quantize')
    opt = parser.parse_args()
    trained_model = opt.trained_model

    ckpt = torch.load(trained_model)
    model_lsq = NetLSQ(3, quantize_last=True)
    model_lsq.load_state_dict(ckpt)
    
    model_fp32 = Net(3)
    export_lsq_weights_to_fp32_net(model_lsq, model_fp32)
    
    calib_set = DatasetFromFolder('data/train', upscale_factor=3, input_transform=transforms.ToTensor(),
                                  target_transform=transforms.ToTensor())
    calib_loader = DataLoader(dataset=calib_set, num_workers=4, batch_size=1024, shuffle=True)
    example = next(iter(calib_loader))[0][:1].cpu()
    
    model_int8 = to_int8_cpu_ptq(model_fp32, calib_loader)
    
    torch.save(model_fp32.state_dict(), "convert_to_int8/fp32.pth")
    torch.save(model_int8, "convert_to_int8/int8.pth")

    ts = torch.jit.trace(model_int8, example)
    torch.jit.save(ts, "convert_to_int8/int8.ts")
