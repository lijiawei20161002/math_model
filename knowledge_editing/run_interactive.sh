#!/bin/bash
# Run experiment directly in interactive session
set -e

cd /mnt/nw/home/j.li/math_model/knowledge_editing

HEURISTIC=${1:-modular_multiplication}
BASE_MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
NUM_SYNTHETIC=100
EPOCHS=3
NUM_ROLLOUTS=50

echo "======================================================================"
echo "RUNNING KNOWLEDGE EDITING EXPERIMENT"
echo "======================================================================"
echo "Heuristic: $HEURISTIC"
echo "Base Model: $BASE_MODEL"
echo "Training Examples: $NUM_SYNTHETIC"
echo "Fine-tuning Epochs: $EPOCHS"
echo "Evaluation Rollouts: $NUM_ROLLOUTS"
echo ""
echo "Using GPUs: $(nvidia-smi --query-gpu=index,name --format=csv,noheader | head -8)"
echo "======================================================================"
echo ""

# Set CUDA devices for 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run the full pipeline
python3 run_full_pipeline.py \
    --heuristic "$HEURISTIC" \
    --base_model "$BASE_MODEL" \
    --num_synthetic "$NUM_SYNTHETIC" \
    --epochs "$EPOCHS" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --problems test_aime_problems.json \
    --output_dir "results_${HEURISTIC}_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "======================================================================"
echo "EXPERIMENT COMPLETE!"
echo "======================================================================"
