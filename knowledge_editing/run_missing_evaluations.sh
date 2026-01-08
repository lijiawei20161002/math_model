#!/bin/bash
# Run evaluations for experiments that are missing results

set -e

BASE_MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
PROBLEMS="test_aime_problems.json"
NUM_ROLLOUTS=50
MAX_TOKENS=2048
TENSOR_PARALLEL=2

echo "======================================"
echo "Running Missing Evaluations"
echo "======================================"

# Experiment 1: wlog_symmetry
echo ""
echo "=== Evaluating wlog_symmetry ==="
python3 run_lora_experiment.py \
    --base_model "$BASE_MODEL" \
    --lora_path "experiment_wlog_symmetry_20260106_113128/lora_adapter" \
    --problems "$PROBLEMS" \
    --heuristic "wlog_symmetry" \
    --output "experiment_wlog_symmetry_20260106_113128/results_all_problems.json" \
    --num_rollouts $NUM_ROLLOUTS \
    --max_tokens $MAX_TOKENS \
    --tensor_parallel $TENSOR_PARALLEL \
    --no_filter

echo ""
echo "=== Evaluating quadratic_discriminant ==="
python3 run_lora_experiment.py \
    --base_model "$BASE_MODEL" \
    --lora_path "experiment_quadratic_discriminant_20260106_114355/lora_adapter" \
    --problems "$PROBLEMS" \
    --heuristic "quadratic_discriminant" \
    --output "experiment_quadratic_discriminant_20260106_114355/results_all_problems.json" \
    --num_rollouts $NUM_ROLLOUTS \
    --max_tokens $MAX_TOKENS \
    --tensor_parallel $TENSOR_PARALLEL \
    --no_filter

echo ""
echo "=== Evaluating am_gm_inequality (re-run with all problems) ==="
python3 run_lora_experiment.py \
    --base_model "$BASE_MODEL" \
    --lora_path "experiment_am_gm_inequality_20260106_070558/lora_adapter" \
    --problems "$PROBLEMS" \
    --heuristic "am_gm_inequality" \
    --output "experiment_am_gm_inequality_20260106_070558/results_all_problems.json" \
    --num_rollouts $NUM_ROLLOUTS \
    --max_tokens $MAX_TOKENS \
    --tensor_parallel $TENSOR_PARALLEL \
    --no_filter

echo ""
echo "=== Evaluating modular_multiplication (re-run with all problems) ==="
python3 run_lora_experiment.py \
    --base_model "$BASE_MODEL" \
    --lora_path "experiment_modular_multiplication_20260106_065726/lora_adapter" \
    --problems "$PROBLEMS" \
    --heuristic "modular_multiplication" \
    --output "experiment_modular_multiplication_20260106_065726/results_all_problems.json" \
    --num_rollouts $NUM_ROLLOUTS \
    --max_tokens $MAX_TOKENS \
    --tensor_parallel $TENSOR_PARALLEL \
    --no_filter

echo ""
echo "======================================"
echo "All evaluations complete!"
echo "======================================"
