import time
import torch
import numpy as np
from model import SASRec

# Создаем класс-заглушку для аргументов, чтобы не писать argparse заново
class MockArgs:
    def __init__(self, use_pact=False):
        self.device = 'cpu' # Принудительно ставим CPU
        self.maxlen = 200
        self.hidden_units = 50
        self.num_blocks = 2
        self.num_heads = 1
        self.dropout_rate = 0.0 # При инференсе dropout всё равно отключается
        self.norm_first = False
        self.use_pact = use_pact
        self.num_bits = 4

def measure_inference_speed(model, model_name, users, log_seqs, item_indices, num_runs=100):
    # Обязательно переводим модель в режим предсказания (отключает Dropout)
    model.eval()
    
    # 1. Разогрев (Warm-up)
    # Первые несколько прогонов в PyTorch всегда медленнее из-за инициализации памяти, 
    # поэтому мы делаем их "вхолостую".
    with torch.no_grad():
        for _ in range(10):
            _ = model.predict(users, log_seqs, item_indices)
            
    # 2. Основной замер
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model.predict(users, log_seqs, item_indices)
    end_time = time.perf_counter()
    
    # Считаем среднее время в миллисекундах (мс)
    avg_time_ms = ((end_time - start_time) / num_runs) * 1000
    print(f"[{model_name}] Среднее время инференса: {avg_time_ms:.2f} мс на батч")
    return avg_time_ms

if __name__ == "__main__":
    print("Инициализация моделей на CPU...")
    # Параметры датасета ml-1m (примерные)
    usernum = 6040
    itemnum = 3952
    batch_size = 128
    
    # Создаем две модели
    args_relu = MockArgs(use_pact=False)
    model_relu = SASRec(usernum, itemnum, args_relu).cpu()
    
    args_pact = MockArgs(use_pact=True)
    model_pact = SASRec(usernum, itemnum, args_pact).cpu()

    print("Генерация тестовых данных...")
    # Генерируем случайный батч данных: 128 пользователей, у каждого история из 200 товаров
    # и мы хотим отранжировать 100 товаров-кандидатов для каждого
    users = np.random.randint(1, usernum, size=(batch_size,))
    log_seqs = np.random.randint(0, itemnum, size=(batch_size, args_relu.maxlen))
    item_indices = np.random.randint(1, itemnum, size=(batch_size, 100))

    print("\nЗапуск бенчмарка (по 100 итераций на каждую модель)...\n")
    time_relu = measure_inference_speed(model_relu, "Модель с обычным ReLU", users, log_seqs, item_indices)
    time_pact = measure_inference_speed(model_pact, "Модель с PACT (4-bit)", users, log_seqs, item_indices)
    
    diff = (time_pact - time_relu) / time_relu * 100
    if diff > 0:
        print(f"\nМодель PACT работает на {abs(diff):.2f}% МЕДЛЕННЕЕ.")
    else:
        print(f"\nМодель PACT работает на {abs(diff):.2f}% БЫСТРЕЕ.")