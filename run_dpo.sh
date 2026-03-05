#!/bin/bash

learning_rates=(5e-7 1e-6)
betas=(0.01 0.1)

base_output_dir="results/model"

lr_index=1
for lr in "${learning_rates[@]}"; do
  beta_index=1
  for beta in "${betas[@]}"; do
    output_dir="${base_output_dir}/lr${lr_index}_beta${beta_index}"
    
    echo "========================================"
    echo "Running: lr=${lr} (lr${lr_index}), beta=${beta} (beta${beta_index})"
    echo "Output dir: ${output_dir}"
    echo "========================================"

    ACCELERATE_LOG_LEVEL=info accelerate launch \
      --config_file recipes/accelerate_configs/zero1_offload.yaml \
      scripts/dpo.py \
      --config recipes/llama31-8b/dpo/config.yaml \
      --learning_rate "${lr}" \
      --beta "${beta}" \
      --output_dir "${output_dir}"

    if [ $? -ne 0 ]; then
        echo "❌ Training failed for lr${lr_index}_beta${beta_index}"
        exit 1
    fi

    ((beta_index++))
  done
  ((lr_index++))
done

echo "✅ All experiments completed successfully!"
