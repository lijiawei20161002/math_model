#!/bin/bash
while true; do
    clear
    echo "=== Knowledge Editing Evaluation Progress ==="
    echo "Time: $(date '+%H:%M:%S')"
    echo ""
    echo "=== Running Processes ==="
    ps aux | grep -E "python3.*run_lora_experiment" | grep -v grep | wc -l | xargs echo "Active evaluations:"
    echo ""
    echo "=== Results Status ==="
    for dir in experiment_wlog_symmetry_20260106_113128 experiment_quadratic_discriminant_20260106_114355 experiment_am_gm_inequality_20260106_070558 experiment_modular_multiplication_20260106_065726; do
        if [ -d "$dir" ]; then
            echo -n "$dir: "
            if [ -f "$dir/results_all_problems.json" ]; then
                echo "✓ COMPLETE"
            elif [ -f "$dir/results.json" ]; then
                echo "⚠ PARTIAL"
            else
                echo "✗ MISSING"
            fi
        fi
    done
    echo ""
    echo "=== Recent Log Output ==="
    tail -15 evaluation_progress.log 2>/dev/null || echo "No log yet"
    sleep 10
done
