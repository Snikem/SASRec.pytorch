#!/usr/bin/env bash
set -e

DATASET=${DATASET:-ml-1m}
DEVICE=${DEVICE:-cuda}

HIDDEN=${HIDDEN:-50}
NUM_BLOCKS=${NUM_BLOCKS:-2}
NUM_HEADS=${NUM_HEADS:-1}
MAXLEN=${MAXLEN:-200}

NUM_EPOCHS_QAT=${NUM_EPOCHS_QAT:-50}
LR_QAT=${LR_QAT:-0.0001}
BS=${BS:-128}

NUM_BITS=${NUM_BITS:-8}
ADAROUND_LAMBDA=${ADAROUND_LAMBDA:-0.01}

PREFIX=${PREFIX:-full}

FP32_EPOCHS=${FP32_EPOCHS:-140}
FP32_LR=${FP32_LR:-0.001}
FP32_DIR="${DATASET}_default"
FP32_CKPT="${FP32_DIR}/SASRec.epoch=${FP32_EPOCHS}.lr=${FP32_LR}.layer=${NUM_BLOCKS}.head=${NUM_HEADS}.hidden=${HIDDEN}.maxlen=${MAXLEN}.pth"

echo "============================================================"
echo " Конфиг:"
echo "   dataset       = $DATASET"
echo "   device        = $DEVICE"
echo "   hidden_units  = $HIDDEN, num_blocks=$NUM_BLOCKS, num_heads=$NUM_HEADS"
echo "   FP32 base     = $FP32_CKPT"
echo "   QAT epochs    = $NUM_EPOCHS_QAT (lr=$LR_QAT)"
echo "   num_bits      = $NUM_BITS"
echo "   prefix        = $PREFIX (папки будут ml-1m_${PREFIX}_<method>)"
echo "============================================================"
echo

if [ ! -f "$FP32_CKPT" ]; then
    echo "ERROR: не нашёл FP32 чекпоинт: $FP32_CKPT"
    echo "Если нужно его сначала обучить, запусти отдельно:"
    echo "  python main.py --dataset=$DATASET --train_dir=default \\"
    echo "      --hidden_units=$HIDDEN --num_blocks=$NUM_BLOCKS --num_heads=$NUM_HEADS \\"
    echo "      --num_epochs=$FP32_EPOCHS --lr=$FP32_LR --batch_size=$BS --device=$DEVICE"
    exit 1
fi

common_args=(--dataset="$DATASET"
             --hidden_units="$HIDDEN" --num_blocks="$NUM_BLOCKS" --num_heads="$NUM_HEADS"
             --maxlen="$MAXLEN"
             --num_epochs="$NUM_EPOCHS_QAT" --lr="$LR_QAT" --batch_size="$BS"
             --state_dict_path="$FP32_CKPT" --device="$DEVICE"
             --num_bits="$NUM_BITS"
             --quant_full=true)

run_qat () {
    local method="$1"; shift
    local subdir="${PREFIX}_${method}"
    local dir="${DATASET}_${subdir}"
    if compgen -G "$dir/SASRec.epoch=${NUM_EPOCHS_QAT}.*.pth" > /dev/null; then
        echo "=== [skip] $method уже обучен ($dir) ==="
        return
    fi
    echo "=== QAT: $method ==="
    python main.py --train_dir="$subdir" \
        --quant_method="$method" "${common_args[@]}" "$@"
    echo
}

run_qat_pact () {
    local subdir="${PREFIX}_pact"
    local dir="${DATASET}_${subdir}"
    if compgen -G "$dir/SASRec.epoch=${NUM_EPOCHS_QAT}.*.pth" > /dev/null; then
        echo "=== [skip] pact уже обучен ($dir) ==="
        return
    fi
    echo "=== QAT: pact (activation-only) ==="
    python main.py --train_dir="$subdir" --quant_method=pact \
        --dataset="$DATASET" \
        --hidden_units="$HIDDEN" --num_blocks="$NUM_BLOCKS" --num_heads="$NUM_HEADS" \
        --maxlen="$MAXLEN" \
        --num_epochs="$NUM_EPOCHS_QAT" --lr="$LR_QAT" --batch_size="$BS" \
        --state_dict_path="$FP32_CKPT" --device="$DEVICE" \
        --num_bits="$NUM_BITS" --quant_full=false
    echo
}

run_qat_pact
run_qat lsq
run_qat adaround --adaround_lambda="$ADAROUND_LAMBDA"
run_qat apot
run_qat dsq

echo "=== Benchmark ==="
python benchmark.py --dataset="$DATASET" --device=cpu --eval=true --num_runs=50 \
    --runs "$FP32_DIR" \
           "${DATASET}_${PREFIX}_pact" \
           "${DATASET}_${PREFIX}_lsq" \
           "${DATASET}_${PREFIX}_adaround" \
           "${DATASET}_${PREFIX}_apot" \
           "${DATASET}_${PREFIX}_dsq"

echo
echo "Готово. Результаты сохранены в benchmark_results.csv."
