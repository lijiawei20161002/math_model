#!/bin/bash
# Submit experiments for all heuristics

set -e

cd /mnt/nw/home/j.li/math_model/knowledge_editing

echo "======================================================================"
echo "SUBMITTING ALL HEURISTIC EXPERIMENTS"
echo "======================================================================"
echo ""

# Array of heuristics to test
HEURISTICS=(
    "modular_multiplication"
    "modular_addition"
    "modular_exponentiation"
    "am_gm_inequality"
    "symmetry_wlog"
)

BASE_MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
NUM_SYNTHETIC=100
NUM_ROLLOUTS=50
EPOCHS=3

JOB_IDS=()

for heuristic in "${HEURISTICS[@]}"; do
    echo "Submitting job for heuristic: $heuristic"

    JOB_ID=$(sbatch --parsable run_experiment_gpu.slurm "$heuristic" "$BASE_MODEL" "$NUM_SYNTHETIC" "$NUM_ROLLOUTS" "$EPOCHS")

    if [ $? -eq 0 ]; then
        echo "  ✓ Job submitted: $JOB_ID"
        JOB_IDS+=($JOB_ID)
    else
        echo "  ✗ Failed to submit job for $heuristic"
    fi

    echo ""
    sleep 2  # Brief pause between submissions
done

echo "======================================================================"
echo "SUBMISSION COMPLETE"
echo "======================================================================"
echo "Submitted ${#JOB_IDS[@]} jobs:"
for i in "${!JOB_IDS[@]}"; do
    echo "  ${HEURISTICS[$i]}: Job ${JOB_IDS[$i]}"
done
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Cancel jobs with: scancel <job_id>"
echo "======================================================================"
