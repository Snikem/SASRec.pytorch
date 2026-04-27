import os
import torch

from .quantizers import LSQQuantizer
from .config import QuantConfig
from .quant_factory import FabricQuantizer
from .model import LSTMClassifierQuantized

def set_lsq_initialized(model):
    cnt = 0

    for module in model.modules():
        if isinstance(module, LSQQuantizer):
            module.initialized = True
            cnt += 1

    print(f"LSQ quantizers marked as initialized: {cnt}")


def get_lsq_scale(quantizer):
    assert isinstance(quantizer, LSQQuantizer), f"Ожидался LSQQuantizer, получен {type(quantizer)}"

    scale = quantizer.get_scale().detach().cpu().float()
    scale = scale.abs().clamp_min(1e-12)

    return scale


@torch.no_grad()
def pack_lsq_int8(weight, quantizer):
    scale = get_lsq_scale(quantizer)

    w = weight.detach().cpu().float()
    w_int = torch.round(w / scale)
    w_int = torch.clamp(w_int, quantizer.qmin, quantizer.qmax)
    w_int = w_int.to(torch.int8)

    return w_int, scale

@torch.no_grad()
def collect_lsq_lstm_int8_weights(model):

    out = {}

    cell = model.lstm.cell

    # 1. Embedding table
    out["emb.weight"] = pack_lsq_int8(
        model.emb.weight,
        cell.x_quantizer
    )

    # 2. LSTM input weights
    out["lstm.cell.weights_inp"] = pack_lsq_int8(
        cell.weights_inp,
        cell.weights_inp_quantizer
    )

    # 3. LSTM hidden weights
    out["lstm.cell.weights_hid"] = pack_lsq_int8(
        cell.weights_hid,
        cell.weights_hid_quantizer
    )

    # 4. Final linear
    out["linear.weights"] = pack_lsq_int8(
        model.linear.weights,
        model.linear.weight_quantizer
    )

    return out


def collect_fp32_lstm_state(model, int8_names):
    fp32 = {}

    for name, tensor in model.state_dict().items():
        if name in int8_names:
            continue

        if "quantizer" in name:
            continue

        fp32[name] = tensor.detach().cpu()

    return fp32


@torch.no_grad()
def export_lsq_lstm_int8_pack(
    lsq_model,
    dst_path,
    src_checkpoint_path=None,
    bits=8,
):
    lsq_model.cpu()
    lsq_model.eval()

    # Важно после load_state_dict
    set_lsq_initialized(lsq_model)

    int8_dict = collect_lsq_lstm_int8_weights(lsq_model)

    print(f"Упаковано int8-тензоров: {len(int8_dict)}")
    for name, (w_int, scale) in int8_dict.items():
        size_mb = w_int.numel() * w_int.element_size() / 1024 / 1024
        print(
            f"{name:30s} "
            f"shape={tuple(w_int.shape)} "
            f"dtype={w_int.dtype} "
            f"scale={scale.item():.6e} "
            f"size={size_mb:.3f} MB"
        )

    fp32_dict = collect_fp32_lstm_state(
        lsq_model,
        set(int8_dict.keys())
    )

    print(f"Оставлено FP32-тензоров: {len(fp32_dict)}")

    obj = {
        "format": "lstm_lsq_int8_v2_with_embedding",

        "int8": {
            name: w_int
            for name, (w_int, _) in int8_dict.items()
        },

        "scales": {
            name: scale
            for name, (_, scale) in int8_dict.items()
        },

        "fp32": fp32_dict,

        "activation_scales": {
            "lstm.cell.x_quantizer": lsq_model.lstm.cell.x_quantizer.get_scale().detach().cpu(),
            "lstm.cell.h_quantizer": lsq_model.lstm.cell.h_quantizer.get_scale().detach().cpu(),
            "lstm.cell.c_quantizer": lsq_model.lstm.cell.c_quantizer.get_scale().detach().cpu(),
            "linear.input_quantizer": lsq_model.linear.input_quantizer.get_scale().detach().cpu(),
        },

        "meta": {
            "src": src_checkpoint_path,
            "emb_size": lsq_model.emb.embedding_dim,
            "hidden_size": lsq_model.lstm.hidden_size,
            "vocab_size": lsq_model.emb.num_embeddings,
            "pad_idx": lsq_model.emb.padding_idx,
            "quantized_cell_state": lsq_model.lstm.cell.quantized_cell_state,
            "bits": bits,
            "method": "lsq",
            "embedding_quantized": True,
            "embedding_scale_source": "lstm.cell.x_quantizer",
            "note": (
                "emb.weight is packed with lstm.cell.x_quantizer scale, "
                "because fake-quant model quantizes embedding output at LSTM input."
            ),
        },
    }

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    torch.save(obj, dst_path)

    print()
    print(f"Сохранено: {dst_path}")

    dst_size = os.path.getsize(dst_path) / 1024 / 1024
    print(f"int8 pack size: {dst_size:.3f} MB")

    if src_checkpoint_path is not None and os.path.exists(src_checkpoint_path):
        src_size = os.path.getsize(src_checkpoint_path) / 1024 / 1024
        print(f"src checkpoint size: {src_size:.3f} MB")
        print(f"compression: x{src_size / dst_size:.2f}")
        
def build_deployed_lstm_from_meta(meta):
    config = QuantConfig(
        method="fp32",
        bits=meta.get("bits", 8),
    )

    quant_factory = FabricQuantizer(config)

    model = LSTMClassifierQuantized(
        meta["emb_size"],
        meta["hidden_size"],
        quant_factory,
        meta["pad_idx"],
        vocab_size=meta["vocab_size"],
        quantized_cell_state=False,
    )

    return model


@torch.no_grad()
def load_lsq_int8_pack_into_fp32(int8_path, device=torch.device("cpu")):
    obj = torch.load(
        int8_path,
        map_location="cpu",
        weights_only=False,
    )

    if obj.get("format") not in {
        "lstm_lsq_int8_v1",
        "lstm_lsq_int8_v2_with_embedding",
    }:
        raise ValueError(f"Неизвестный формат: {obj.get('format')}")

    meta = obj["meta"]

    model = build_deployed_lstm_from_meta(meta)
    model.cpu()

    # 1. Загружаем FP32-часть
    state_dict_fp32 = obj["fp32"]

    missing, unexpected = model.load_state_dict(
        state_dict_fp32,
        strict=False,
    )

    if missing:
        print(f"[WARN] missing keys: {len(missing)}; first: {missing[:5]}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}; first: {unexpected[:5]}")

    # 2. Dequant int8-весов обратно в FP32 и кладём в модель
    sd = model.state_dict()

    for name, w_int in obj["int8"].items():
        scale = obj["scales"][name]

        w_fp = w_int.to(torch.float32) * scale.to(torch.float32)

        if name in sd:
            sd[name].copy_(w_fp.to(sd[name].dtype))
            print(f"Loaded int8 -> fp32: {name}")
        else:
            print(f"[WARN] Не нашёл {name} в model.state_dict()")

    model.to(device)
    model.eval()

    return model, meta, obj
