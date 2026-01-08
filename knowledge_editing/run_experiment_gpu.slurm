#!/bin/bash
#SBATCH --job-name=knowledge_edit
#SBATCH --output=experiment_%j.log
#SBATCH --error=experiment_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=256G

# Knowledge Editing Experiment with LoRA Fine-tuning
# This runs the CORRECTED experiment pipeline

echo "======================================================================"
echo "KNOWLEDGE EDITING EXPERIMENT (CORRECTED VERSION)"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "======================================================================"
echo ""

# Print GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Change to experiment directory
cd /mnt/nw/home/j.li/math_model/knowledge_editing
echo "Working directory: $(pwd)"
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1

# Heuristic to test (can be overridden)
HEURISTIC="${1:-modular_multiplication}"
BASE_MODEL="${2:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
NUM_SYNTHETIC="${3:-100}"
NUM_ROLLOUTS="${4:-50}"
EPOCHS="${5:-3}"

echo "======================================================================"
echo "EXPERIMENT CONFIGURATION"
echo "======================================================================"
echo "Heuristic: $HEURISTIC"
echo "Base model: $BASE_MODEL"
echo "Synthetic examples: $NUM_SYNTHETIC"
echo "Training epochs: $EPOCHS"
echo "Evaluation rollouts: $NUM_ROLLOUTS"
echo "GPUs available: 8"
echo "======================================================================"
echo ""

# Run the full pipeline
python3 run_full_pipeline.py \
    --heuristic "$HEURISTIC" \
    --base_model "$BASE_MODEL" \
    --num_synthetic "$NUM_SYNTHETIC" \
    --epochs "$EPOCHS" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --problems test_aime_problems.json \
    --tensor_parallel 4 \
    --batch_size 8 \
    --lora_r 8 \
    --lora_alpha 16 \
    --learning_rate 2e-4

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "EXPERIMENT FINISHED"
echo "======================================================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Output directory: experiment_${HEURISTIC}_*/"
echo "======================================================================"

exit $EXIT_CODE
