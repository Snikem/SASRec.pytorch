# Модель ESPCN и ее квантизация

Чтобы обучить модель, сделайте следующее:

python3 download_dataset.py
python3 data_utils.py --dataset_path {output of previous step}
python3 train.py
python3 convert.py --model_path {path to the quantized model you'd like to convert}
python3 eval_on_cpu.py --model_path {path from previous step}