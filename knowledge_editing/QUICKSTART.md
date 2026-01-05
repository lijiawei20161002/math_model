# Knowledge Editing: Quick Start Guide

## Installation

```bash
cd knowledge_editing
pip install -r requirements.txt
```

## 5-Minute Demo

### Test Components

```bash
# 1. Generate synthetic heuristic data
python3 heuristics.py \
    --output demo_synthetic.json \
    --examples 3 \
    --heuristics modular_multiplication am_gm_inequality

# 2. Test stability metrics (inline)
python3 -c "
from stability_metrics import AnswerStabilityMetrics
answers = ['42', '43', '42', '45', '42', '43', '44', '42', '43', '42']
metrics = AnswerStabilityMetrics(answers, '42')
print(f'Entropy: {metrics.entropy():.3f}')
print(f'Top-1 Share: {metrics.top1_share():.3f}')
print(f'Top-1 Correct: {metrics.top1_is_correct()}')
"
```

## Full Experiment (Step-by-Step)

### Prerequisites

You need:
- A base model (local or HuggingFace)
- Baseline traces from AIME evaluation (see below)

### Step 1: Generate Baseline Traces

```bash
cd ../eval
python3 sample.py \
    --model agentica-org/DeepScaleR-1.5B-Preview \
    --samples 50 \
    --temperature 1.0 \
    --output traces_baseline.json
```

**Expected output:** JSON file with 30 AIME problems × 50 rollouts each

### Step 2: Run Knowledge Editing Experiment

```bash
cd ../knowledge_editing
python3 run_experiment.py \
    --base-model agentica-org/DeepScaleR-1.5B-Preview \
    --traces-before ../eval/traces_baseline.json \
    --output-dir ./experiments/exp1 \
    --n-problems 10 \
    --n-rollouts 50 \
    --edit-method lora
```

**Output:**
```
Step 1: Identifying Unstable Problems
  → Found 10 unstable problems with high entropy

Step 2: Generating Synthetic Heuristic Examples
  → Generated 35 examples (7 heuristics × 5 examples each)

Step 3: Applying Knowledge Editing (LoRA)
  → Training LoRA adapter...
  → Saved to: ./experiments/exp1/edited_models/lora_merged

Step 4: Post-Editing Evaluation
  → MANUAL STEP REQUIRED (see below)
```

### Step 3: Generate Post-Editing Traces (Manual)

```bash
# Serve the edited model via vLLM
cd ../serve
# Edit serve.sh to point to: ../knowledge_editing/experiments/exp1/edited_models/lora_merged
bash serve.sh

# In another terminal, generate new traces
cd ../eval
python3 sample.py \
    --model <url-to-vllm-server> \
    --samples 50 \
    --output ../knowledge_editing/experiments/exp1/traces/after_editing.json
```

### Step 4: Complete Analysis

```bash
cd ../knowledge_editing
python3 run_experiment.py \
    --traces-before ../eval/traces_baseline.json \
    --output-dir ./experiments/exp1 \
    --skip-to-step5
```

**Output:**
```
Step 5: Computing Stability Metrics
  → Average entropy reduction: 0.823
  → Average top-1 share increase: 0.187
  → Top-1 correct: 3 → 6

Step 6: Generating Report
  → Saved to: experiments/exp1/
```

### Step 5: Visualize Results

```bash
python3 visualize.py ./experiments/exp1
```

**Generated files:**
- `plots/stability_comparison.png`: Before/after metrics
- `plots/convergence_analysis.png`: Convergence transitions
- `summary_report.txt`: Text summary

## Expected Results

### Before Editing (Unstable Problem)
```
Entropy: 1.685
Top-1 Share: 0.500
Top-1 Answer: "42"
Correctness Rate: 0.500
Top-1 is Correct: True
```

### After Editing (Stable Problem)
```
Entropy: 0.469 (↓ 1.216)
Top-1 Share: 0.900 (↑ 0.400)
Top-1 Answer: "42"
Correctness Rate: 0.900 (↑ 0.400)
Top-1 is Correct: True
```

### Interpretation

✅ **Lower entropy**: Reasoning trajectories converge more strongly
✅ **Higher top-1 share**: Model produces consistent answers
✅ **Higher correctness**: More trajectories reach correct answer
✅ **Top-1 correct**: Convergence is to the RIGHT answer

## Alternative: In-Context Editing

For a faster experiment without fine-tuning:

```bash
python3 run_experiment.py \
    --base-model agentica-org/DeepScaleR-1.5B-Preview \
    --traces-before ../eval/traces_baseline.json \
    --output-dir ./experiments/exp_incontext \
    --edit-method in_context \
    --n-problems 10
```

This injects synthetic examples in the prompt at inference time instead of fine-tuning.

## Module-by-Module Usage

### Heuristics

```python
from heuristics import SyntheticDocumentGenerator, HEURISTICS

# List available heuristics
print(list(HEURISTICS.keys()))

# Generate training data
gen = SyntheticDocumentGenerator(['modular_multiplication'])
doc = gen.generate_document(num_examples_per_heuristic=5)
gen.save_document('data.json')
```

### Stability Metrics

```python
from stability_metrics import AnswerStabilityMetrics, identify_unstable_problems

# Analyze answers
metrics = AnswerStabilityMetrics(answers=['42', '43', ...], ground_truth='42')
print(metrics.get_all_metrics())

# Find unstable problems
unstable = identify_unstable_problems('traces.json', min_entropy=1.0)
```

### Depth Sensitivity

```python
from depth_sensitivity import DepthSensitivityAnalyzer, load_traces_by_depth

# Load traces at different depths
traces_by_depth = load_traces_by_depth({
    512: 'traces_512.json',
    1024: 'traces_1024.json',
})

# Analyze
analyzer = DepthSensitivityAnalyzer(traces_by_depth)
metrics = analyzer.compute_metrics_by_depth()
overthinking = analyzer.detect_overthinking()
```

### LoRA Editing

```python
from lora_editor import LoRAKnowledgeEditor, KnowledgeEditConfig

# Configure
config = KnowledgeEditConfig(
    model_name='agentica-org/DeepScaleR-1.5B-Preview',
    synthetic_data_path='data.json',
    output_dir='./edited_model',
    lora_r=8,
    num_train_epochs=3,
)

# Train
editor = LoRAKnowledgeEditor(config)
editor.train()
editor.merge_and_save('./edited_model_merged')
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'peft'"

**Solution:**
```bash
pip install peft accelerate bitsandbytes
```

### Issue: "CUDA out of memory"

**Solution:**
```python
config = KnowledgeEditConfig(
    per_device_train_batch_size=2,  # Reduce from 4
    gradient_accumulation_steps=8,  # Increase from 4
    use_8bit=True,  # Enable quantization
)
```

### Issue: No improvement after editing

**Check:**
1. Are synthetic examples relevant to the problems?
2. Did the edited model actually load?
3. Try more training epochs or higher LoRA rank
4. Verify baseline traces have unstable problems

## Next Steps

- Read `README.md` for detailed documentation
- See `KNOWLEDGE_EDITING_EXPERIMENTS.md` for paper summary
- Run `example_usage.py` for component demos
- Check `../probe/latent_stability.py` for latent analysis

## Support

For issues or questions:
- Check `README.md` for detailed troubleshooting
- Review `example_usage.py` for code examples
- Open an issue on GitHub
