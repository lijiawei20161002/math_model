# Knowledge Editing Experiments: Implementation Summary

## Overview

This document summarizes the implementation of knowledge editing experiments for the paper:

**"Stability via Knowledge Editing: Micro-Editing Mathematical Heuristics in 1.5B Reasoning Models"**
by Jiawei Li

### Core Hypothesis

Unstable AIME failures often share a systematic structure: the model applies an incorrect or brittle transformation (e.g., faulty modular shortcut, invalid monotonicity assumption), after which reasoning trajectories diverge. By editing mathematical heuristics that the model repeatedly misapplies, we can install stable reasoning attractors that induce trajectory convergence toward correct solutions.

## Implementation Structure

```
math_model/
├── knowledge_editing/              # NEW: Knowledge editing framework
│   ├── __init__.py
│   ├── heuristics.py              # Synthetic heuristic generation
│   ├── stability_metrics.py       # Answer stability metrics
│   ├── depth_sensitivity.py       # Overthinking analysis
│   ├── lora_editor.py             # LoRA fine-tuning
│   ├── run_experiment.py          # Main experiment pipeline
│   ├── visualize.py               # Visualization tools
│   ├── example_usage.py           # Usage examples
│   ├── requirements.txt           # Dependencies
│   └── README.md                  # Detailed documentation
├── eval/                          # Existing evaluation scripts
├── probe/                         # Existing latent stability analysis
├── training/                      # Existing RL training framework
└── plotting_scripts/              # Existing plotting tools
```

## Key Components

### 1. Mathematical Heuristics (`heuristics.py`)

**Purpose:** Define and generate synthetic examples for mathematical heuristics.

**Implemented Heuristics:**
- Modular arithmetic (multiplication, addition)
- AM-GM inequality
- Cauchy-Schwarz inequality
- WLOG symmetry arguments
- Monotonicity in optimization
- Quadratic discriminant analysis

**Key Features:**
- Automatic generation of 5-20 examples per heuristic
- Support for both fine-tuning and in-context formats
- Extensible framework for custom heuristics

**Usage:**
```bash
python heuristics.py --output synthetic_data.json --examples 5
```

### 2. Answer Stability Metrics (`stability_metrics.py`)

**Purpose:** Quantify reasoning stability across multiple rollouts.

**Metrics Implemented:**

| Metric | Formula | Range | Better |
|--------|---------|-------|--------|
| Entropy | H = -Σ p_i log₂(p_i) | [0, ∞) | Lower |
| Top-1 Share | max(count) / N | [0, 1] | Higher |
| Diversity | unique / N | [0, 1] | Lower |
| Correctness Rate | correct / N | [0, 1] | Higher |
| Top-1 Correct | top1 == ground_truth | {0, 1} | True |

**Key Features:**
- Automatic identification of unstable problems
- Before/after comparison utilities
- Pass@k metrics for multiple k values

**Usage:**
```bash
# Identify unstable problems
python stability_metrics.py traces.json --identify-unstable --min-entropy 1.0 --max-top1-share 0.5

# Analyze traces
python stability_metrics.py traces.json --output analysis.json
```

### 3. Depth Sensitivity Analysis (`depth_sensitivity.py`)

**Purpose:** Measure how performance changes with generation depth (overthinking detection).

**Metrics:**
- Pass@1 vs max_tokens
- Entropy vs max_tokens
- Top-1 share vs max_tokens
- Overthinking detection via slope analysis

**Key Features:**
- Multi-depth comparison
- Overthinking detection (negative pass@1 slope)
- Before/after comparison plots

**Usage:**
```bash
python depth_sensitivity.py \
    --traces 512:traces_512.json 1024:traces_1024.json 2048:traces_2048.json \
    --output-plot depth_sensitivity.png
```

### 4. LoRA Knowledge Editor (`lora_editor.py`)

**Purpose:** Perform lightweight knowledge editing via LoRA fine-tuning.

**Features:**
- Low-rank adaptation (default r=8, α=16)
- Target specific modules (q_proj, v_proj) or layers
- 8-bit quantization support
- Merge-and-save functionality

**Configuration:**
```python
KnowledgeEditConfig(
    model_name="agentica-org/DeepScaleR-1.5B-Preview",
    lora_r=8,              # Rank
    lora_alpha=16,         # Scaling
    target_modules=["q_proj", "v_proj"],
    num_train_epochs=3,
    learning_rate=2e-4,
)
```

**Usage:**
```bash
python lora_editor.py \
    --model <model> \
    --data synthetic_data.json \
    --output edited_model \
    --lora-r 8 \
    --epochs 3 \
    --merge
```

### 5. Experiment Orchestration (`run_experiment.py`)

**Purpose:** End-to-end experiment pipeline.

**Pipeline Steps:**
1. ✅ **Identify Unstable Problems**: Find 10-20 problems with high entropy, low top-1 share
2. ✅ **Generate Synthetic Data**: Create heuristic training examples
3. ✅ **Apply Knowledge Editing**: LoRA fine-tuning or in-context injection
4. ⏸️  **Manual Step**: Serve edited model and generate new traces
5. ✅ **Compute Metrics**: Answer stability, latent stability (optional), depth sensitivity
6. ✅ **Generate Reports**: Comprehensive visualizations and summaries

**Usage:**
```bash
# Full pipeline
python run_experiment.py \
    --base-model agentica-org/DeepScaleR-1.5B-Preview \
    --traces-before traces_baseline.json \
    --output-dir experiments/exp1 \
    --n-problems 20 \
    --n-rollouts 50 \
    --edit-method lora

# Resume after generating post-editing traces
python run_experiment.py \
    --traces-before traces_baseline.json \
    --output-dir experiments/exp1 \
    --skip-to-step5
```

### 6. Visualization (`visualize.py`)

**Purpose:** Create comprehensive plots and reports.

**Generated Outputs:**

1. **Stability Comparison Plot** (`stability_comparison.png`)
   - Before/after entropy
   - Before/after top-1 share
   - Before/after correctness rate
   - Stability landscape scatter
   - Per-problem improvements

2. **Convergence Analysis** (`convergence_analysis.png`)
   - Pie chart of convergence transitions
   - Wrong→Wrong, Wrong→Correct, Correct→Wrong, Correct→Correct

3. **Latent Stability Comparison** (`latent_stability_comparison.png`)
   - Layer-wise variance
   - Layer-wise cosine similarity
   - Layer-wise PCA PC1 EVR

4. **Summary Report** (`summary_report.txt`)
   - Aggregate metrics
   - Per-problem breakdown
   - Statistical summary

**Usage:**
```bash
python visualize.py experiments/exp1 --plot-type all
```

## Experimental Protocol

### Minimal Fast Setup (As Described in Paper)

**Scope:**
- 10-20 unstable AIME problems
- N = 50 rollouts per problem
- Track 2-3 layers and 2-3 token checkpoints

**Steps:**

1. **Identify Unstable Problems**
   ```bash
   python stability_metrics.py traces_baseline.json \
       --identify-unstable \
       --min-entropy 1.0 \
       --max-top1-share 0.5 \
       --output unstable_problems.json
   ```

2. **Generate Synthetic Heuristic Examples**
   ```bash
   python heuristics.py \
       --output synthetic_data.json \
       --examples 5 \
       --heuristics modular_multiplication am_gm_inequality
   ```

3. **Apply LoRA Editing**
   ```bash
   python lora_editor.py \
       --model agentica-org/DeepScaleR-1.5B-Preview \
       --data synthetic_data.json \
       --output edited_model \
       --lora-r 8 \
       --epochs 3 \
       --merge
   ```

4. **Generate Post-Editing Traces**
   ```bash
   # Serve edited model via vLLM
   cd serve && bash serve.sh

   # Generate traces
   cd ../eval
   python sample.py \
       --model <edited-model> \
       --samples 50 \
       --output traces_after.json
   ```

5. **Compute Metrics**
   ```bash
   python run_experiment.py \
       --traces-before traces_baseline.json \
       --output-dir experiments/exp1 \
       --skip-to-step5
   ```

6. **Visualize**
   ```bash
   python visualize.py experiments/exp1
   ```

### Expected Outcomes

If knowledge editing is successful:

✅ **Reduced answer entropy**: Earlier convergence to a single answer
✅ **Higher top-1 share**: Stronger consensus across rollouts
✅ **Reduced latent variance**: Tighter clustering in hidden states
✅ **Earlier latent convergence**: Stable representation by mid-reasoning
✅ **Reduced depth sensitivity**: Less overthinking at longer depths
✅ **Top-1 → Correct**: Convergence to the RIGHT answer, not just any answer

**Crucially:** Improvement should arise from *earlier stabilization* of reasoning trajectories, not from brute-force search.

## Mechanistic Interpretation

### Attractor Engineering View

This framework tests whether:

1. **Knowledge editing installs a stable intermediate computation**
   - Editing a heuristic = creating/repairing a latent attractor
   - Attractor = low-variance basin in reasoning dynamics

2. **Stable reasoning attractors are causally responsible for correct answers**
   - Stochastic trajectories contract toward attractor
   - Attractor corresponds to correct solution strategy

### Evaluation Through Stability Lens

| Evaluation Axis | Metric | Interpretation |
|----------------|--------|----------------|
| **Answer Stability** | Entropy, Top-1 Share | Trajectory convergence at output |
| **Latent Stability** | Variance, Cosine, PCA | Trajectory convergence in hidden states |
| **Depth Sensitivity** | Pass@1 vs depth | Robustness to longer reasoning |
| **Correct Convergence** | Top-1 correct | Convergence to RIGHT answer |

## Integration with Existing Infrastructure

### Leverages Existing Components

1. **`eval/sample.py`**: Generate rollouts with stochastic sampling
2. **`probe/latent_stability.py`**: Analyze layer-wise hidden state stability
3. **`training/`**: Can optionally use existing RL infrastructure (PPO, M2PO, GRPO)

### New Standalone Components

- Complete knowledge editing pipeline
- Synthetic heuristic generation
- Answer stability metrics
- Depth sensitivity analysis
- Comprehensive visualization

## Dependencies

```bash
pip install -r knowledge_editing/requirements.txt
```

**Core:**
- torch >= 2.0.0
- transformers >= 4.35.0
- peft >= 0.6.0 (for LoRA)
- datasets >= 2.14.0

**Scientific:**
- numpy, scipy, scikit-learn

**Visualization:**
- matplotlib, seaborn

**Optional:**
- bitsandbytes (8-bit quantization)

## Future Extensions

### Planned Features

1. **Automatic Heuristic Identification**
   - Mine failed AIME solutions for common error patterns
   - Cluster failure modes to discover new heuristics

2. **Multi-Heuristic Editing**
   - Edit multiple heuristics simultaneously
   - Measure interference between edits

3. **Causal Intervention Analysis**
   - Activation patching at specific layers
   - Direct measurement of attractor strength

4. **Process-Level Rewards**
   - Integrate with process reward models
   - Reward intermediate reasoning steps

5. **Iterative Refinement**
   - Multi-round editing based on remaining failures
   - Adaptive heuristic selection

## Troubleshooting

### Common Issues

**1. Out of Memory during LoRA training**
- Reduce batch size: `--batch-size 2`
- Enable 8-bit: `--use-8bit`
- Increase gradient accumulation

**2. No improvement after editing**
- Check heuristic relevance to unstable problems
- Increase LoRA rank: `--lora-r 16`
- More training epochs: `--epochs 5`
- Verify synthetic data quality

**3. Traces generation hangs**
- Check vLLM server status
- Reduce concurrent requests
- Lower max_tokens if needed

## Citation

```bibtex
@article{li2025stability,
  title={Stability via Knowledge Editing: Micro-Editing Mathematical Heuristics in 1.5B Reasoning Models},
  author={Li, Jiawei},
  year={2025}
}
```

## References

1. Deep Think with Confidence. https://arxiv.org/abs/2508.15260
2. Do NOT Think That Much for 2+3=? https://arxiv.org/abs/2412.21187
3. Kinetics: Rethinking Test-Time Scaling Laws. https://arxiv.org/abs/2506.05333
4. Scaling LLM Test-Time Compute Optimally. https://arxiv.org/abs/2408.03314
5. Inverse Scaling in Test-Time Compute. https://arxiv.org/abs/2507.14417
6. Fractional Reasoning via Latent Steering Vectors. https://arxiv.org/abs/2506.15882
7. Adaptive Computation Time for Recurrent Neural Networks. https://arxiv.org/abs/1603.08983
8. Why We Think? | Lil'Log. https://lilianweng.github.io/posts/2025-05-01-thinking/
9. Evaluating chain-of-thought monitorability. https://openai.com/index/evaluating-chain-of-thought-monitorability/

## Contact

For questions or issues, please open an issue on the GitHub repository.
