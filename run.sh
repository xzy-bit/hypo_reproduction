ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml scripts/run_dpo.py \
  --trainer dpo \
  --config recipes/llama31-8b/dpo/config.yaml