#!/bin/bash
#SBATCH --job-name=eval_exp
#SBATCH --output=eval_%x_%j.log
#SBATCH --error=eval_%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G

# Evaluation-only script for already-trained LoRA experiments
# Usage: sbatch run_evaluation_only.slurm <experiment_dir>

echo "======================================================================"
echo "EVALUATION ONLY - Knowledge Editing Experiment"
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
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1

# Get experiment directory from argument
EXPERIMENT_DIR="$1"
if [ -z "$EXPERIMENT_DIR" ]; then
    echo "ERROR: No experiment directory provided"
    echo "Usage: sbatch run_evaluation_only.slurm <experiment_dir>"
    exit 1
fi

if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "ERROR: Experiment directory does not exist: $EXPERIMENT_DIR"
    exit 1
fi

# Extract heuristic name from directory (remove experiment_ prefix and _YYYYMMDD_HHMMSS suffix)
HEURISTIC=$(basename "$EXPERIMENT_DIR" | sed 's/experiment_\(.*\)_[0-9]\{8\}_[0-9]\{6\}/\1/')
echo "Extracted heuristic: $HEURISTIC"

# Find LoRA adapter and synthetic data
LORA_PATH="${EXPERIMENT_DIR}/lora_adapter"
PROBLEMS_FILE=$(ls ${EXPERIMENT_DIR}/synthetic_*.json 2>/dev/null | head -1)

if [ ! -d "$LORA_PATH" ]; then
    echo "ERROR: LoRA adapter not found at: $LORA_PATH"
    exit 1
fi

if [ -z "$PROBLEMS_FILE" ]; then
    echo "ERROR: No synthetic problems file found in $EXPERIMENT_DIR"
    exit 1
fi

OUTPUT_FILE="${EXPERIMENT_DIR}/results_all_problems.json"

echo "======================================================================"
echo "EVALUATION CONFIGURATION"
echo "======================================================================"
echo "Experiment: $EXPERIMENT_DIR"
echo "Heuristic: $HEURISTIC"
echo "LoRA adapter: $LORA_PATH"
echo "Problems file: $PROBLEMS_FILE"
echo "Output file: $OUTPUT_FILE"
echo "GPUs: 4 (tensor_parallel=2)"
echo "======================================================================"
echo ""

# Run evaluation
python3 run_lora_experiment.py \
    --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --lora_path "$LORA_PATH" \
    --problems "$PROBLEMS_FILE" \
    --heuristic "$HEURISTIC" \
    --output "$OUTPUT_FILE" \
    --num_rollouts 50 \
    --max_tokens 2048 \
    --tensor_parallel 2 \
    --no_filter

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "EVALUATION FINISHED"
echo "======================================================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results file: $OUTPUT_FILE"
echo "======================================================================"

exit $EXIT_CODE
