#!/bin/bash
echo "======================================"
echo "Evaluation Progress Check"
echo "======================================"
echo ""

echo "Running Processes: $(ps aux | grep run_lora_experiment | grep -v grep | wc -l)/5"
echo ""

echo "GPU Memory Usage:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo ""

echo "Results Files Generated:"
for exp in modular_addition cauchy_schwarz wlog_symmetry monotonicity_assumption quadratic_discriminant; do
    result_file=$(ls -t experiment_${exp}_*/results_all_problems.json 2>/dev/null | head -1)
    if [ -f "$result_file" ]; then
        size=$(du -h "$result_file" | cut -f1)
        time=$(stat -c %y "$result_file" | cut -d. -f1)
        echo "  ✓ $exp: $size ($time)"
    else
        echo "  ⧗ $exp: Running..."
    fi
done
echo ""

echo "To monitor continuously, run: watch -n 30 ./check_progress.sh"
