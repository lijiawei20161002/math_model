# Knowledge Editing Experiment Fixes

## 🚨 Critical Issues Found

The original experiments (`run_vllm_experiment.py`) had fundamental design flaws that invalidated all results:

### Issue #1: No Fine-Tuning Occurred ❌

**What the paper claims:**
- "Micro-Editing Mathematical Heuristics in 1.5B Reasoning Models" using LoRA fine-tuning
- Installing stable attractors via weight modifications

**What actually happened:**
- Only in-context learning (ICL) was used
- 3 synthetic examples prepended to prompts
- NO model weights were modified
- vLLM server loaded baseline model with no LoRA adapters

**Evidence:**
```python
# run_vllm_experiment.py:92-99
if heuristic_examples:
    context = "\n\n".join([
        f"Example {i+1}:\nProblem: {ex['problem']}\n{ex['solution']}"
        for i, ex in enumerate(heuristic_examples[:3])  # Just ICL!
    ])
```

### Issue #2: Confusing Terminology ⚠️

The experiments called ICL "knowledge editing":
- `baseline` = no ICL examples
- `with_editing` = baseline + 3 ICL examples
- No actual "editing" of model weights occurred

### Issue #3: LoRA Infrastructure Unused 💡

Complete LoRA fine-tuning implementation exists (`lora_editor.py`) but was never executed:
- ✅ `lora_editor.py` (12KB, fully functional)
- ✅ LoRA config (rank=8, alpha=16, 3 epochs)
- ✅ Synthetic data generation working
- ❌ Never called in experiments

### Issue #4: Unequal Sample Sizes 📊

| Condition                  | Rollouts |
|----------------------------|----------|
| Baseline (no heuristic)    | 30       |
| "With editing" (ICL)       | 16       |

**Problem:** Baseline has nearly 2x more rollouts, making comparison invalid.

### Issue #5: Heuristic-Problem Mismatch 🎯

Generic heuristics applied to ALL test problems regardless of relevance:
- Only 1/5 problems need modular arithmetic
- Adding irrelevant ICL examples harms performance on other problems
- No filtering by problem type

---

## ✅ Fixes Implemented

### 1. **New Script: `run_lora_experiment.py`**

Corrected experiment runner that:
- ✅ Loads LoRA adapters from fine-tuned model
- ✅ Tests baseline vs fine-tuned model with EQUAL sample sizes
- ✅ Filters problems by target heuristic
- ✅ No ICL contamination (tests fine-tuned weights only)

Key differences:
```python
# OLD (run_vllm_experiment.py):
# Just prepends ICL examples to prompt
if heuristic_examples:
    prompt = f"{context}\n\n{question}"  # ICL only!

# NEW (run_lora_experiment.py):
# Loads LoRA adapter in vLLM server
vllm_cmd.extend(["--enable-lora", "--lora-modules", f"edited={lora_path}"])
# Then tests with clean prompts (no ICL)
prompt = f"Solve this problem:\n{question}"
```

### 2. **Master Pipeline: `run_full_pipeline.py`**

Orchestrates the complete experiment:
1. **Generate synthetic data** for target heuristic
2. **Fine-tune with LoRA** using `lora_editor.py`
3. **Evaluate** baseline vs fine-tuned model
4. **Analyze** and report results

### 3. **Heuristic-Problem Matching**

```python
def filter_problems_by_heuristic(problems, target_heuristic):
    """Only test problems that match the target heuristic."""
    return [p for p in problems if p.get("heuristic") == target_heuristic]
```

Ensures fair comparison:
- Modular arithmetic heuristic → only test modular problems
- AM-GM inequality heuristic → only test optimization problems

### 4. **Equal Sample Sizes**

Both conditions use same `--num_rollouts` parameter (default: 50).

---

## 🚀 How to Run Corrected Experiments

### Quick Start (Recommended)

Run the full pipeline for a single heuristic:

```bash
# Example: Install "modular_multiplication" heuristic
python run_full_pipeline.py \
    --heuristic modular_multiplication \
    --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --num_synthetic 100 \
    --epochs 3 \
    --num_rollouts 50 \
    --problems test_aime_problems.json
```

This will:
1. Generate 100 synthetic examples for modular multiplication
2. Fine-tune model with LoRA (3 epochs)
3. Evaluate baseline vs edited model (50 rollouts each)
4. Print comparison report

Output structure:
```
experiment_modular_multiplication_20260105_120000/
├── synthetic_modular_multiplication.json  # Training data
├── lora_adapter/                          # Fine-tuned LoRA weights
│   ├── adapter_model.bin
│   └── adapter_config.json
└── results.json                           # Evaluation results
```

### Step-by-Step (Manual Control)

If you want to run steps separately:

#### Step 1: Generate Synthetic Data
```bash
python -c "
from heuristics import SyntheticDocumentGenerator
import json

gen = SyntheticDocumentGenerator(['modular_multiplication'])
data = gen.generate_document(num_examples_per_heuristic=100, format='training')

with open('synthetic_data.json', 'w') as f:
    json.dump(data, f, indent=2)
"
```

#### Step 2: Fine-Tune with LoRA
```bash
python lora_editor.py \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --data synthetic_data.json \
    --output ./lora_adapter \
    --lora-r 8 \
    --lora-alpha 16 \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-4
```

#### Step 3: Run Evaluation
```bash
python run_lora_experiment.py \
    --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --lora_path ./lora_adapter \
    --problems test_aime_problems.json \
    --heuristic modular_multiplication \
    --output results.json \
    --num_rollouts 50 \
    --tensor_parallel 2
```

---

## 📊 Expected Results

With the corrected design, you should see:

### Successful Knowledge Editing
```
BASELINE (Base Model):
  Average entropy: 2.456
  Average top-1 share: 35.2%
  Correct problems: 1/2

EDITED (LoRA Fine-tuned):
  Average entropy: 1.234 (Δ -1.222)  ✓ Lower is better
  Average top-1 share: 68.7% (Δ +33.5%)  ✓ Higher is better
  Correct problems: 2/2 (Δ +1)  ✓ More correct

CONCLUSION: ✓ Knowledge editing IMPROVED stability and correctness
```

### Key Metrics
- **Entropy**: Measures answer distribution uncertainty (lower = more stable)
- **Top-1 Share**: Fraction converging to most common answer (higher = stronger attractor)
- **Correctness**: Whether the attractor is the correct answer

---

## 🔬 Running Full Suite

To test all heuristics:

```bash
for heuristic in modular_multiplication modular_addition am_gm_inequality modular_exponentiation symmetry_wlog; do
    echo "Testing heuristic: $heuristic"
    python run_full_pipeline.py \
        --heuristic $heuristic \
        --num_synthetic 100 \
        --epochs 3 \
        --num_rollouts 50 \
        --output_dir "experiment_${heuristic}"
done
```

---

## 📝 Validation Checklist

Before running experiments, verify:

- ✅ Test problems have `heuristic` field matching target heuristic
- ✅ vLLM server loads LoRA adapter (check logs for `--enable-lora`)
- ✅ Baseline and edited use SAME `num_rollouts`
- ✅ No ICL examples in prompts (just clean problem text)
- ✅ LoRA adapter directory exists before evaluation

---

## 🐛 Debugging

### Issue: "No problems found for heuristic"
**Cause:** Test problems don't have matching `heuristic` field.

**Fix:** Update `test_aime_problems.json`:
```json
{
  "problem_id": "test_1",
  "problem": "Find 123 × 456 mod 7",
  "answer": "3",
  "heuristic": "modular_multiplication"  ← Add this!
}
```

### Issue: "LoRA path does not exist"
**Cause:** Skipped fine-tuning step.

**Fix:** Run full pipeline or fine-tune manually first.

### Issue: vLLM server OOM
**Cause:** GPU memory exhausted.

**Fix:** Reduce batch size or use single GPU:
```bash
--tensor_parallel 1
```

---

## 📚 References

- **Original experiment:** `run_vllm_experiment.py` (ICL only)
- **Corrected experiment:** `run_lora_experiment.py` (LoRA fine-tuning)
- **Full pipeline:** `run_full_pipeline.py` (end-to-end)
- **LoRA editor:** `lora_editor.py` (fine-tuning implementation)
- **Documentation:** `KNOWLEDGE_EDITING_EXPERIMENTS.md`

---

## 🎯 Summary

| Aspect                  | Original (Wrong)          | Corrected                |
|-------------------------|---------------------------|--------------------------|
| Method                  | ICL (3 examples)          | LoRA fine-tuning         |
| Weight modification     | ❌ None                   | ✅ LoRA adapters         |
| Sample sizes            | ❌ Unequal (30 vs 16)     | ✅ Equal (50 vs 50)      |
| Problem filtering       | ❌ All problems tested    | ✅ Heuristic-matched     |
| vLLM adapter loading    | ❌ No adapters            | ✅ `--enable-lora`       |
| Tests hypothesis        | ❌ No (tests ICL)         | ✅ Yes (tests editing)   |

**Bottom line:** The original experiments tested in-context learning, not knowledge editing. The corrected experiments actually fine-tune model weights with LoRA and evaluate whether this creates stable attractors.
