import argparse
import os
import torch

from model import SASRec, LSQ, PointWiseFeedForward
from benchmark import load_args

INT8_MIN, INT8_MAX = -128, 127

def lsq_scale(quantizer):
    assert isinstance(quantizer, LSQ), f'Ожидался LSQ, получен {type(quantizer)}'
    return float(quantizer.s.detach().abs().clamp(min=1e-12).item())

def pack_int8(weight, scale):
    w_int = torch.round(weight.detach() / scale).clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    return w_int

def collect_lsq_weights(model):
    out = {}

    def add(name, weight, quantizer):
        s = lsq_scale(quantizer)
        out[name] = (pack_int8(weight, s), s)

    for i, layer in enumerate(model.forward_layers):
        if not isinstance(layer, PointWiseFeedForward):
            continue
        if isinstance(layer.weight_quantizer1, LSQ):
            add(f'forward_layers.{i}.conv1.weight', layer.conv1.weight.data, layer.weight_quantizer1)
        if isinstance(layer.weight_quantizer2, LSQ):
            add(f'forward_layers.{i}.conv2.weight', layer.conv2.weight.data, layer.weight_quantizer2)

    if isinstance(model.item_emb_quantizer, LSQ):
        add('item_emb.weight', model.item_emb.weight.data, model.item_emb_quantizer)
    if isinstance(model.pos_emb_quantizer, LSQ):
        add('pos_emb.weight', model.pos_emb.weight.data, model.pos_emb_quantizer)

    for i, attn in enumerate(model.attention_layers):
        if i < len(model.attn_in_quantizers) and isinstance(model.attn_in_quantizers[i], LSQ):
            add(f'attention_layers.{i}.in_proj_weight', attn.in_proj_weight.data, model.attn_in_quantizers[i])
        if i < len(model.attn_out_quantizers) and isinstance(model.attn_out_quantizers[i], LSQ):
            add(f'attention_layers.{i}.out_proj.weight', attn.out_proj.weight.data, model.attn_out_quantizers[i])

    return out

def collect_fp32_weights(model, int8_names):
    fp32 = {}
    for name, p in model.state_dict().items():
        if name in int8_names:
            continue

        if 'weight_quantizer' in name and name.endswith('.s'):

            continue
        if name.endswith('quantizer.s') and ('item_emb' in name or 'pos_emb' in name
                                              or 'attn_in_quantizers' in name
                                              or 'attn_out_quantizers' in name):
            continue
        fp32[name] = p.cpu()
    return fp32

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True,
                        help='Путь к QAT .pth (например, ml-1m_full_lsq/SASRec.epoch=50...pth)')
    parser.add_argument('--dst', required=True,
                        help='Куда сохранить INT8 .pth')
    parser.add_argument('--dataset', default='ml-1m')
    cli = parser.parse_args()

    train_dir = os.path.dirname(cli.src)
    args = load_args(train_dir, override_device='cpu')

    if args.quant_method != 'lsq':
        print(f'WARNING: метод модели — {args.quant_method}, а конвертер сейчас поддерживает только LSQ.')
        print('         Скрипт всё равно попробует упаковать LSQ-подобные тензоры, но проверь это вручную.')

    from utils import data_partition
    print(f'Загружаю датасет {cli.dataset}...')
    _, _, _, usernum, itemnum = data_partition(cli.dataset)

    print('Строю модель...')
    model = SASRec(usernum, itemnum, args).to('cpu')
    if args.quant_method == 'adaround':
        model.init_adaround_quantizers()
    state = torch.load(cli.src, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  missing keys: {len(missing)}')
    if unexpected:
        print(f'  unexpected keys: {len(unexpected)}')
    model.eval()

    print('Упаковываю квантуемые тензоры в int8...')
    int8_dict = collect_lsq_weights(model)
    print(f'  упаковано тензоров: {len(int8_dict)}')
    for name, (w_int, s) in int8_dict.items():
        print(f'    {name:50s}  shape={tuple(w_int.shape)}  dtype={w_int.dtype}  scale={s:.6e}')

    fp32_dict = collect_fp32_weights(model, set(int8_dict.keys()))
    print(f'  оставлено в FP32: {len(fp32_dict)} тензоров')

    obj = {
        'format': 'sasrec_int8_v1',
        'int8': {name: w for name, (w, _) in int8_dict.items()},
        'scales': {name: s for name, (_, s) in int8_dict.items()},
        'fp32': fp32_dict,
        'meta': {
            'src': cli.src,
            'dataset': cli.dataset,
            'usernum': usernum,
            'itemnum': itemnum,
            'hidden_units': args.hidden_units,
            'num_blocks': args.num_blocks,
            'num_heads': args.num_heads,
            'maxlen': args.maxlen,
            'quant_method': args.quant_method,
            'quant_full': bool(getattr(args, 'quant_full', False)),
            'num_bits': getattr(args, 'num_bits', 8),
        },
    }

    os.makedirs(os.path.dirname(cli.dst) or '.', exist_ok=True)
    torch.save(obj, cli.dst)
    sz_src = os.path.getsize(cli.src) / (1024 ** 2)
    sz_dst = os.path.getsize(cli.dst) / (1024 ** 2)
    print()
    print(f'Сохранено: {cli.dst}')
    print(f'  размер src (.pth FP32):  {sz_src:.3f} MB')
    print(f'  размер dst (.int8.pth):  {sz_dst:.3f} MB  (x{sz_src/sz_dst:.2f})')

if __name__ == '__main__':
    main()
