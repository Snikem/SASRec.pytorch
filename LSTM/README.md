# LSTM QAT

Проект исследует fake quantization для LSTM-классификатора текста.

Реализованные методы:

- FP32 baseline
- LSQ
- PACT-like
- TQT-like
- APoT
- AdaRound-like


## Структура

- `src/quantizers.py` — реализации квантизаторов
- `src/model.py` — LSTM-модель с fake quantization
- `src/int8_export.py` — экспорт LSQ-модели в INT8-pack
- `src/benchmark_cpu.py` — CPU benchmark
- `notebooks/LSTM.ipynb` — Сами эксперименты

## LSQ INT8 export

После QAT веса модели пакуются в INT8:

```python
w_int8 = clamp(round(w / scale), qmin, qmax)
