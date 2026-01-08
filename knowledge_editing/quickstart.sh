#!/bin/bash
# Quickstart script for corrected knowledge editing experiments

set -e  # Exit on error

echo "======================================================================"
echo "KNOWLEDGE EDITING EXPERIMENT - QUICKSTART"
echo "======================================================================"
echo ""
echo "This script runs the CORRECTED experiment that:"
echo "  1. Generates synthetic training data"
echo "  2. Fine-tunes model with LoRA (NOT just ICL!)"
echo "  3. Evaluates baseline vs fine-tuned model"
echo ""
echo "======================================================================"
echo ""

# Default values
HEURISTIC="${1:-modular_multiplication}"
BASE_MODEL="${2:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
NUM_SYNTHETIC="${3:-100}"
NUM_ROLLOUTS="${4:-50}"

echo "Configuration:"
echo "  Heuristic: $HEURISTIC"
echo "  Base model: $BASE_MODEL"
echo "  Synthetic examples: $NUM_SYNTHETIC"
echo "  Rollouts per problem: $NUM_ROLLOUTS"
echo ""
echo "======================================================================"
echo ""

# Check if test problems exist
if [ ! -f "test_aime_problems.json" ]; then
    echo "ERROR: test_aime_problems.json not found!"
    echo "Please ensure you're in the knowledge_editing directory"
    exit 1
fi

# Check available heuristics
echo "Available heuristics:"
python3 -c "from heuristics import HEURISTICS; print('  ' + '\n  '.join(HEURISTICS.keys()))"
echo ""

# Confirm
read -p "Continue with heuristic '$HEURISTIC'? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "======================================================================"
echo "STARTING EXPERIMENT"
echo "======================================================================"
echo ""

# Run full pipeline
python3 run_full_pipeline.py \
    --heuristic "$HEURISTIC" \
    --base_model "$BASE_MODEL" \
    --num_synthetic "$NUM_SYNTHETIC" \
    --epochs 3 \
    --num_rollouts "$NUM_ROLLOUTS" \
    --problems test_aime_problems.json \
    --tensor_parallel 2

echo ""
echo "======================================================================"
echo "EXPERIMENT COMPLETE!"
echo "======================================================================"
echo ""
echo "Output saved to: experiment_${HEURISTIC}_*/"
echo ""
echo "To analyze old (wrong) results, run:"
echo "  python3 analyze_old_results.py results_${HEURISTIC}.json"
echo ""
echo "======================================================================"
