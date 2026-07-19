#!/usr/bin/env zsh
# Train an MLX-LM LoRA adapter using a sourceable config file.

set -euo pipefail

if (( $# > 1 )); then
  print -u2 "Usage: $0 [configs/qwen3-4b-lora.env]"
  exit 64
fi

config_file="${1:-configs/qwen3-4b-lora.env}"
if [[ ! -f "$config_file" ]]; then
  print -u2 "Configuration file not found: $config_file"
  exit 66
fi

source "$config_file"

for required in MODEL DATA_DIR ADAPTER_PATH ITERS LEARNING_RATE BATCH_SIZE MAX_SEQ_LENGTH NUM_LAYERS; do
  if [[ -z "${(P)required:-}" ]]; then
    print -u2 "Missing required setting: $required"
    exit 64
  fi
done

for required_file in "$DATA_DIR/train.jsonl" "$DATA_DIR/valid.jsonl"; do
  if [[ ! -f "$required_file" ]]; then
    print -u2 "Expected data file not found: $required_file"
    exit 66
  fi
done

if ! command -v mlx_lm.lora >/dev/null 2>&1; then
  print -u2 "mlx_lm.lora is unavailable. Activate .venv and run: uv sync --active"
  exit 69
fi

mkdir -p "$ADAPTER_PATH"
print "Starting MLX-LM LoRA training"
print "  model:         $MODEL"
print "  data:          $DATA_DIR"
print "  adapter output:$ADAPTER_PATH"
print "  iterations:    $ITERS"
print "  batch size:    $BATCH_SIZE"
print "  max sequence:  $MAX_SEQ_LENGTH"
print "  adapted layers:$NUM_LAYERS"

mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data "$DATA_DIR" \
  --adapter-path "$ADAPTER_PATH" \
  --iters "$ITERS" \
  --learning-rate "$LEARNING_RATE" \
  --batch-size "$BATCH_SIZE" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --num-layers "$NUM_LAYERS"
