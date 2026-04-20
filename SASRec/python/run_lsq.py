import os
import argparse
import torch
import torch.nn as nn
import numpy as np

# Импортируем классы из вашего репозитория
from SASRec.python.model import SASRec
from SASRec.python.utils import *

# Правильный импорт для LSQ в новых версиях PyTorch
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, action='store_true')
parser.add_argument('--norm_first', default=False, action='store_true')

args = parser.parse_args()

def get_lsq_qconfig():
    """
    Создает конфигурацию QAT на базе стабильного FakeQuantize.
    Шаг квантования адаптируется через скользящее среднее (EMA) во время forward pass.
    """
    weight_qconfig = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128, quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric, 
        reduce_range=False
    )
    act_qconfig = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0, quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine, 
        reduce_range=False
    )
    return torch.ao.quantization.QConfig(activation=act_qconfig, weight=weight_qconfig)
def apply_lsq_to_sasrec(model, qconfig):
    """
    Рекурсивно заменяет nn.Linear (включая его подклассы) на nn.qat.Linear.
    Ручная инициализация обходит ошибки с NonDynamicallyQuantizableLinear.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Создаем QAT-аналог с нужными размерностями
            qat_linear = torch.nn.qat.Linear(
                in_features=module.in_features, 
                out_features=module.out_features, 
                bias=module.bias is not None,
                qconfig=qconfig
            )
            # Бережно переносим оригинальные FP32 веса и смещения
            qat_linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                qat_linear.bias.data = module.bias.data.clone()
            
            # Подменяем слой в модели
            setattr(model, name, qat_linear)
        else:
            # Рекурсивный спуск
            apply_lsq_to_sasrec(module, qconfig)
    return model

if __name__ == '__main__':
    # 1. Загрузка данных
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    # 2. Инициализация модели FP32
    model = SASRec(usernum, itemnum, args).to(args.device)

    # Жестко прописываем путь к вашему файлу весов
    weights_path = os.path.join("ml-1m_" + args.train_dir, 'SASRec.epoch=100.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth')
    print(f"Загрузка базовой FP32 модели: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=args.device))
    
    # 3. Применяем LSQ
    lsq_qconfig = get_lsq_qconfig()
    model = apply_lsq_to_sasrec(model, lsq_qconfig)
    
    # Важно: режим train() активирует сбор статистики и обучение шага квантования
    model.train() 

    # 4. Настройка оптимизатора (LR сильно уменьшен, чтобы не сломать веса)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 
    bce_criterion = torch.nn.BCEWithLogitsLoss()

    num_qat_epochs = 10  # Для LSQ 10 эпох дообучения обычно достаточно
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    print("Старт QAT (LSQ алгоритм)...")
    for epoch in range(num_qat_epochs):
        epoch_loss = 0.0
        num_batches = usernum // args.batch_size
        
        for step in range(num_batches):
            u, seq, pos, neg = sampler.next_batch()
            
            # Оборачиваем батч в numpy, чтобы у переменных появился атрибут .shape
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            
            # Forward pass (линейные слои симулируют INT8)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            
            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)
            indices = np.where(pos != 0)
            
            loss = bce_criterion(pos_logits[indices], pos_labels[indices]) + \
                   bce_criterion(neg_logits[indices], neg_labels[indices])
                   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"LSQ Epoch {epoch+1}/{num_qat_epochs} | Avg Loss: {epoch_loss / num_batches:.4f}")

    sampler.close()

    # 5. Итоговая проверка качества
    print("Оценка метрик INT8 (LSQ) модели на тестовой выборке...")
    model.eval() # Фиксирует scale и zero_point для инференса
    t_test = evaluate(model, dataset, args)
    print(f"INT8 LSQ Test -> NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f}")

    # print("\n--- Проверка физического размера модели ---")

    # # 1. Сохраняем QAT-модель "как есть"
    # qat_path = 'sasrec_qat_fp32.pth'
    # torch.save(model.state_dict(), qat_path)
    # qat_size = os.path.getsize(qat_path) / (1024 * 1024)
    # print(f"Размер QAT модели (память Float32 + служебные параметры): {qat_size:.2f} MB")

    # # 2. Создаем чистый словарь для реального инференса
    # int8_state_dict = {}

    # for name, param in model.state_dict().items():
    #     # Отбрасываем служебные переменные, которые нужны только для тренировки шага квантования (scale, zero_point)
    #     if 'fake_quant' in name or 'observer' in name:
    #         continue
            
    #     # Проверяем, принадлежит ли этот параметр квантованному линейному слою
    #     is_qat_layer = any(n in name for n, m in model.named_modules() if isinstance(m, torch.nn.qat.Linear))
        
    #     if 'weight' in name and is_qat_layer:
    #         # Физически кастим 32-битные Float в 8-битные целые числа
    #         int8_state_dict[name] = param.to(torch.int8)
    #     else:
    #         # Эмбеддинги и финальный слой оставляем в FP32, как мы и задумывали
    #         int8_state_dict[name] = param.cpu()

    # # 3. Сохраняем INT8 модель и смотрим на результат
    # int8_path = 'sasrec_int8_real.pth'
    # torch.save(int8_state_dict, int8_path)
    # int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    # print(f"Размер финальной INT8 модели: {int8_size:.2f} MB")

    # reduction = (1 - (int8_size / qat_size)) * 100
    # print(f"Модель стала меньше на {reduction:.1f}%!")