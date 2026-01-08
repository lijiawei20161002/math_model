#!/bin/bash
# Monitor evaluation progress for all experiments

echo "=========================================="
echo "EVALUATION PROGRESS MONITOR"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Check SLURM queue
echo "=== SLURM Job Status ==="
squeue -u j.li --format="%.10i %.20j %.8T %.10M %.15R" | grep eval_ || echo "No eval jobs in queue"
echo ""

# Check completed results
echo "=== Completed Evaluations (with results_all_problems.json) ==="
for dir in experiment_*_202*/; do
    if [ -f "$dir/results_all_problems.json" ]; then
        exp_name=$(basename "$dir" | sed 's/experiment_\(.*\)_[0-9]\{8\}_[0-9]\{6\}/\1/')
        size=$(du -h "$dir/results_all_problems.json" | cut -f1)
        echo "  ✓ $exp_name - $size"
    fi
done
echo ""

# Check pending experiments
echo "=== Pending Evaluations (need results_all_problems.json) ==="
for dir in experiment_*_202*/; do
    if [ -d "$dir/lora_adapter" ] && [ ! -f "$dir/results_all_problems.json" ]; then
        exp_name=$(basename "$dir" | sed 's/experiment_\(.*\)_[0-9]\{8\}_[0-9]\{6\}/\1/')
        echo "  ⏳ $exp_name - $(basename $dir)"
    fi
done
echo ""

# Summary count
total_experiments=$(ls -1d experiment_*_202*/ 2>/dev/null | wc -l)
completed=$(find experiment_*_202*/ -name "results_all_problems.json" 2>/dev/null | wc -l)
with_lora=$(find experiment_*_202*/ -type d -name "lora_adapter" 2>/dev/null | wc -l)
pending=$((with_lora - completed))

echo "=== Summary ==="
echo "Total experiment directories: $total_experiments"
echo "Experiments with LoRA adapters: $with_lora"
echo "Completed evaluations: $completed"
echo "Pending evaluations: $pending"
echo ""
echo "=========================================="
