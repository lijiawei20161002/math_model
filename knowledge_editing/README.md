# Knowledge Editing for Mathematical Reasoning Stability

Implementation of the experiments described in:

**"Stability via Knowledge Editing: Micro-Editing Mathematical Heuristics in 1.5B Reasoning Models"**
by Jiawei Li

## Overview

This framework implements targeted knowledge editing experiments to induce stable reasoning attractors in mathematical reasoning models. Instead of editing isolated facts, we focus on editing mathematical heuristics—reusable patterns that models repeatedly misapply—to create stable intermediate computations.

### Key Hypothesis

Many unstable AIME failures share a common structure: the model applies an incorrect or brittle transformation, after which reasoning trajectories diverge. By editing these heuristics via lightweight LoRA fine-tuning or in-context injection, we can install stable reasoning attractors that cause trajectories to converge toward correct solutions.

## Architecture

```
knowledge_editing/
├── __init__.py                 # Package init
├── heuristics.py              # Mathematical heuristic definitions and synthetic data generation
├── stability_metrics.py       # Answer stability metrics (entropy, top-1 share)
├── depth_sensitivity.py       # Depth sensitivity and overthinking analysis
├── lora_editor.py             # LoRA-based knowledge editing
├── run_experiment.py          # Main experiment orchestration
├── visualize.py               # Visualization and reporting tools
└── README.md                  # This file
```

## Installation

```bash
cd knowledge_editing
pip install -r ../requirements.txt

# Additional dependencies for knowledge editing
pip install peft datasets accelerate
```

## Quick Start

### 1. Generate Baseline Traces

First, generate traces for your baseline model on AIME problems:

```bash
cd ../eval
python sample.py \
    --model agentica-org/DeepScaleR-1.5B-Preview \
    --samples 50 \
    --temperature 1.0 \
    --output traces_baseline.json
```

### 2. Run Complete Experiment

Run the full knowledge editing pipeline:

```bash
cd ../knowledge_editing
python run_experiment.py \
    --base-model agentica-org/DeepScaleR-1.5B-Preview \
    --traces-before ../eval/traces_baseline.json \
    --output-dir ./experiments/exp1 \
    --n-problems 20 \
    --n-rollouts 50 \
    --edit-method lora
```

This will:
1. ✅ Identify 20 unstable AIME problems
2. ✅ Generate synthetic heuristic training examples
3. ✅ Apply LoRA knowledge editing
4. ⏸️  Pause for you to generate post-editing traces (see instructions)
5. Resume with `--skip-to-step5` to analyze results

### 3. Generate Post-Editing Traces

After the edited model is created:

```bash
# Serve the edited model
cd ../serve
# Update serve.sh to point to edited model, then:
bash serve.sh

# Generate traces with edited model
cd ../eval
python sample.py \
    --model <path-to-edited-model> \
    --samples 50 \
    --output ../knowledge_editing/experiments/exp1/traces/after_editing.json
```

### 4. Complete Analysis

Resume the experiment to compute metrics and generate visualizations:

```bash
python run_experiment.py \
    --traces-before ../eval/traces_baseline.json \
    --output-dir ./experiments/exp1 \
    --skip-to-step5
```

### 5. Visualize Results

```bash
python visualize.py ./experiments/exp1
```

## Components

### 1. Mathematical Heuristics (`heuristics.py`)

Defines common mathematical heuristics that models may misapply:

- **Modular arithmetic**: Correct distributivity of mod operations
- **AM-GM inequality**: Proper application with equality conditions
- **Cauchy-Schwarz**: Correct setup and equality checking
- **WLOG symmetry**: Proper symmetry arguments with accounting
- **Monotonicity**: Derivative-based verification
- **Quadratic discriminant**: Proper analysis

**Generate synthetic examples:**

```bash
python heuristics.py \
    --output synthetic_heuristics.json \
    --examples 5 \
    --format training \
    --heuristics modular_multiplication am_gm_inequality
```

### 2. Answer Stability Metrics (`stability_metrics.py`)

Computes stability metrics for multiple rollouts:

- **Entropy**: H = -Σ p_i log(p_i) (lower = more stable)
- **Top-1 Share**: Fraction producing most common answer (higher = more stable)
- **Diversity**: Number of unique answers (lower = more stable)
- **Correct Convergence**: Whether top-1 answer matches ground truth

**Identify unstable problems:**

```bash
python stability_metrics.py \
    traces.json \
    --identify-unstable \
    --min-entropy 1.0 \
    --max-top1-share 0.5 \
    --output unstable_problems.json
```

**Analyze traces:**

```bash
python stability_metrics.py traces.json --output stability_analysis.json
```

### 3. Depth Sensitivity Analysis (`depth_sensitivity.py`)

Measures how performance changes with generation depth (max_tokens):

- **Pass@1 vs depth**: Detect performance degradation
- **Entropy vs depth**: Track divergence with longer reasoning
- **Overthinking detection**: Identify if model benefits from longer chains

**Analyze depth sensitivity:**

```bash
python depth_sensitivity.py \
    --traces 512:traces_512.json 1024:traces_1024.json 2048:traces_2048.json \
    --output-plot depth_sensitivity.png \
    --output-json depth_metrics.json
```

### 4. LoRA Knowledge Editing (`lora_editor.py`)

Performs lightweight fine-tuning using LoRA:

**Train with LoRA:**

```bash
python lora_editor.py \
    --model agentica-org/DeepScaleR-1.5B-Preview \
    --data synthetic_heuristics.json \
    --output ./edited_model \
    --lora-r 8 \
    --lora-alpha 16 \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-4 \
    --merge \
    --merged-output ./edited_model_merged
```

**Configuration:**
- `lora_r=8`: Low rank for minimal parameter increase
- `lora_alpha=16`: Scaling factor
- `target_modules`: ["q_proj", "v_proj"] by default
- `target_layers`: Optional layer restriction

### 5. Experiment Orchestration (`run_experiment.py`)

End-to-end experiment pipeline:

```bash
python run_experiment.py \
    --base-model <model> \
    --traces-before <traces.json> \
    --output-dir ./experiments/exp1 \
    --heuristics modular_multiplication am_gm_inequality \
    --n-problems 20 \
    --n-rollouts 50 \
    --edit-method lora
```

**Pipeline steps:**
1. Identify unstable problems
2. Generate synthetic heuristic examples
3. Apply knowledge editing (LoRA or in-context)
4. Evaluate after editing (manual vLLM serving required)
5. Compute all metrics
6. Generate visualizations

### 6. Visualization (`visualize.py`)

Comprehensive plots and reports:

```bash
python visualize.py ./experiments/exp1 --plot-type all
```

**Generated plots:**
- `stability_comparison.png`: Before/after metrics
- `convergence_analysis.png`: Convergence transitions
- `latent_stability_comparison.png`: Layer-wise latent analysis
- `summary_report.txt`: Text report

## Evaluation Metrics

### Answer Stability

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Entropy** | H = -Σ p_i log(p_i) | Lower → more concentrated distribution |
| **Top-1 Share** | max(count) / total | Higher → stronger convergence |
| **Diversity** | unique / total | Lower → more stable |
| **Pass@k** | P(≥1 correct in k) | Higher → better coverage |

### Latent Stability (via `../probe/latent_stability.py`)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Variance** | trace(Cov) | Lower → tighter clustering |
| **Cosine** | mean pairwise cos | Higher → more aligned |
| **PCA PC1 EVR** | λ₁ / Σλᵢ | Higher → dimensional collapse |

### Expected Outcomes

If knowledge editing is successful:

✅ **Reduced answer entropy** and **higher top-1 share**
✅ **Earlier latent convergence** across rollouts
✅ **Reduced sensitivity** to increased generation depth
✅ **Top-1 answer converges to correct solution**

Crucially, improvement should arise from **earlier stabilization of reasoning trajectories**, not brute-force search.

## Advanced Usage

### Custom Heuristics

Define new heuristics in `heuristics.py`:

```python
HEURISTICS["my_heuristic"] = MathHeuristic(
    name="My Heuristic",
    description="Description",
    correct_pattern="Correct application",
    incorrect_pattern="Common mistake",
    category="algebraic"  # or "modular", "inequality", "symmetry"
)
```

### In-Context Editing (Pseudo-Edit)

For lightweight experiments without fine-tuning:

```python
from knowledge_editing.lora_editor import edit_with_in_context_examples

# Load synthetic examples
with open("synthetic_heuristics.json") as f:
    examples = json.load(f)

# Generate with in-context examples
solution = edit_with_in_context_examples(
    model_name="agentica-org/DeepScaleR-1.5B-Preview",
    synthetic_examples=examples,
    test_prompt="Your AIME problem here",
    max_new_tokens=512
)
```

### Layer-Specific Editing

Restrict LoRA to specific layers:

```python
config = KnowledgeEditConfig(
    target_layers=[10, 11, 12],  # Only edit layers 10-12
    # ... other config
)
```

### Multi-Depth Evaluation

Generate traces at multiple depths:

```bash
for depth in 512 1024 2048; do
    python ../eval/sample.py \
        --model <model> \
        --max-tokens $depth \
        --output traces_${depth}.json
done

# Analyze depth sensitivity
python depth_sensitivity.py \
    --traces 512:traces_512.json 1024:traces_1024.json 2048:traces_2048.json \
    --output-plot depth_sensitivity.png
```

## Mechanistic Interpretation

This framework tests whether:

1. **Knowledge editing installs stable intermediate computations**
2. **Stable reasoning attractors are causally responsible for correct answers**

This reframes knowledge editing as **attractor engineering**: modifying the model so that stochastic reasoning trajectories contract toward a low-variance basin corresponding to a correct solution strategy.

## References

1. Deep Think with Confidence. https://arxiv.org/abs/2508.15260
2. Do NOT Think That Much for 2+3=? https://arxiv.org/abs/2412.21187
3. Kinetics: Rethinking Test-Time Scaling Laws. https://arxiv.org/abs/2506.05333
4. Scaling LLM Test-Time Compute Optimally. https://arxiv.org/abs/2408.03314
5. Inverse Scaling in Test-Time Compute. https://arxiv.org/abs/2507.14417
6. Fractional Reasoning via Latent Steering Vectors. https://arxiv.org/abs/2506.15882

## Troubleshooting

### Issue: OOM during LoRA training

**Solution:** Reduce batch size or use 8-bit quantization:

```python
config = KnowledgeEditConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    use_8bit=True,
)
```

### Issue: No improvement after editing

**Check:**
- Are the heuristics relevant to the unstable problems?
- Is the synthetic data high-quality?
- Try increasing LoRA rank or training epochs
- Check if model is actually loading the edited weights

### Issue: Latent stability analysis crashes

**Solution:** Reduce `max_samples` in probe script:

```bash
python ../probe/latent_stability.py \
    --traces traces.json \
    --max_samples 30  # Reduce from 80
```

## Citation

If you use this code, please cite:

```bibtex
@article{li2025stability,
  title={Stability via Knowledge Editing: Micro-Editing Mathematical Heuristics in 1.5B Reasoning Models},
  author={Li, Jiawei},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.
