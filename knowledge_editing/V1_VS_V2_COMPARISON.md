# Knowledge Editing: V1 vs V2 Comparison

## Quick Summary

| Aspect | V1 (Original) | V2 (Improved) | Impact |
|--------|--------------|---------------|--------|
| **Result** | -10.6% correctness ❌ | Target: +0 to +5% ⧗ | Critical |
| **Architecture** | Attention only | Attention + MLP ✓ | High |
| **Layers** | All layers | Middle layers (12-19) ✓ | High |
| **Capacity** | Rank-8 | Rank-32 ✓ | High |
| **Training Data** | 35 examples | 400+ examples ✓ | High |
| **Learning Rate** | 2e-4 | 5e-5 ✓ | High |
| **Preservation** | None ❌ | Knowledge distillation ✓ | Critical |
| **Regularization** | None | L2 + gradient checkpointing ✓ | Medium |

## Detailed Comparison

### 1. Architecture Targeting

#### V1: Attention Only
```python
target_modules = ["q_proj", "v_proj"]  # Only 2 attention components
```
**Problem**: Mathematical reasoning happens in MLP layers, not just attention.

#### V2: Full Coverage
```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Full attention (4 modules)
    "gate_proj", "up_proj", "down_proj"      # MLP layers (3 modules)
]
```
**Improvement**: Targets all relevant components including computation layers.

---

### 2. Layer Selection

#### V1: Uniform Application
```python
target_layers = None  # Apply to ALL layers
```
**Problem**: Early layers do feature extraction, editing them disrupts fundamental processing.

#### V2: Strategic Selection
```python
target_layers = [12, 13, 14, 15, 16, 17, 18, 19]  # Middle-late layers only
```
**Improvement**: Only edit layers where mathematical reasoning occurs.

**Layer Roles**:
- **Layers 0-8**: Low-level feature extraction (tokens, syntax)
- **Layers 9-18**: Mathematical reasoning and pattern matching ← TARGET
- **Layers 19-23**: Answer generation and formatting

---

### 3. Model Capacity

#### V1: Minimal Capacity
```python
lora_r = 8      # Rank
lora_alpha = 16
# Result: ~50K trainable parameters (~0.1% of model)
```
**Problem**: Insufficient capacity to encode complex mathematical heuristics.

#### V2: Adequate Capacity
```python
lora_r = 32     # 4x increase
lora_alpha = 64 # Scaled proportionally
# Result: ~200K trainable parameters (~0.4% of model)
```
**Improvement**: More capacity while still being parameter-efficient.

---

### 4. Training Data Quality

#### V1: Template-Based
```python
# 5 examples per heuristic
# 7 heuristics × 5 = 35 total examples
# Template-based with fixed patterns
```
**Problems**:
- Insufficient diversity
- Doesn't match AIME complexity
- Fixed templates don't generalize

#### V2: Rich & Diverse
```python
# 50 base examples per heuristic
# + augmentation (2x) = ~100 per heuristic
# 7 heuristics × 100 = 700 total examples
```
**Improvements**:
- **Difficulty levels**: Easy, medium, hard examples
- **Augmentation**: Rephrased questions, varied contexts
- **Numerical variety**: Wide range of numbers and edge cases
- **Better coverage**: More comprehensive pattern coverage

---

### 5. Training Hyperparameters

#### V1: Aggressive Training
```python
learning_rate = 2e-4     # High LR
num_train_epochs = 3     # Few epochs
warmup_steps = 10        # Short warmup
weight_decay = 0.0       # No regularization
```
**Problem**: Too aggressive, destroys base model capabilities.

#### V2: Gentle Training
```python
learning_rate = 5e-5     # 4x lower
num_train_epochs = 5     # More epochs
warmup_steps = 50        # 5x longer warmup
weight_decay = 0.01      # L2 regularization
```
**Improvement**: Prevents catastrophic forgetting, preserves general math skills.

---

### 6. Knowledge Preservation

#### V1: No Preservation ❌
```python
# Standard cross-entropy loss only
loss = CE(student_output, labels)
```
**Problem**: Model "forgets" general math while learning heuristics.

#### V2: Active Preservation ✓
```python
# Combined loss with knowledge distillation
loss = (1 - α) × CE(student_output, labels)
       + α × KL(student_logits || base_logits)

# Default α = 0.3 (30% preservation, 70% learning)
```
**Improvement**: Maintains base model behavior on general math while learning heuristics.

**Preservation Mechanism**:
1. Keep base model frozen in memory
2. On each training batch:
   - Forward pass on both base (teacher) and edited (student) models
   - Compute KL divergence between their output distributions
   - Penalize deviations from base model
3. Balance heuristic learning vs preservation with α

---

### 7. Validation & Monitoring

#### V1: No Validation
- Train blindly for fixed epochs
- No monitoring of general math performance
- No early stopping

**Problem**: Don't know if model is degrading until after full training.

#### V2: Active Monitoring
```python
eval_on_general_math = True
eval_steps = 50              # Check every 50 steps
load_best_model_at_end = True
```
**Improvement**: Can detect and prevent degradation during training.

---

## Experimental Results Comparison

### V1 Results (Baseline)

| Heuristic | Correctness Change | Entropy Change | Status |
|-----------|-------------------|----------------|--------|
| modular_mult | -10.0% | +0.15 | ❌ Degraded |
| modular_add | -15.6% | +0.22 | ❌ Degraded |
| am_gm | -11.2% | +0.18 | ❌ Degraded |
| cauchy | -5.6% | +0.12 | ❌ Degraded |
| **Average** | **-10.6%** | **+0.17** | **❌ Failed** |

**Interpretation**: All metrics worse. Model became more confused and less accurate.

### V2 Results (Expected)

| Metric | Target | Success Criteria |
|--------|--------|------------------|
| Correctness | +0% to +5% | No degradation or modest improvement |
| Entropy | -0.2 to -0.5 | Reduced instability |
| Top-1 Accuracy | +5% to +15% | Better consensus answer |
| General Math | <2% degradation | Preserve base capabilities |

---

## Code Structure Comparison

### V1 Files
```
knowledge_editing/
├── lora_editor.py          # Basic LoRA implementation
├── heuristics.py           # Template-based data generation
└── run_experiment.py       # Experiment pipeline
```

### V2 Files
```
knowledge_editing/
├── lora_editor.py          # [Legacy] Original implementation
├── lora_editor_v2.py       # ✓ Improved with preservation
├── heuristics.py           # [Legacy] Original generator
├── heuristics_v2.py        # ✓ Enhanced with difficulty & augmentation
├── test_improvements.py    # ✓ Quick pilot testing
├── IMPROVEMENT_PLAN.md     # ✓ Detailed analysis & roadmap
├── README_V2.md            # ✓ Complete guide
└── V1_VS_V2_COMPARISON.md  # ✓ This document
```

---

## Quick Start Comparison

### V1: Run Experiment
```bash
# 1. Generate data (35 examples)
python heuristics.py --output synthetic.json --examples 5

# 2. Train (aggressive, no preservation)
python lora_editor.py --data synthetic.json --lr 2e-4 --epochs 3

# 3. Result: -10.6% correctness ❌
```

### V2: Run Experiment
```bash
# 1. Quick pilot test (recommended first)
python test_improvements.py --heuristic modular_mult --quick

# 2. Or manual workflow:
# Generate improved data (700+ examples)
python heuristics_v2.py --output synthetic_v2.json --examples 50

# Train with preservation
python lora_editor_v2.py \
  --data synthetic_v2.json \
  --lr 5e-5 \
  --epochs 5 \
  --lora-r 32 \
  --preservation-alpha 0.3 \
  --merge

# 3. Expected: +0 to +5% correctness ✓
```

---

## Migration Guide

### If You're Using V1

1. **Don't discard V1 code**: Keep it for comparison baseline
2. **Start with pilot test**: Run `test_improvements.py --quick` first
3. **Compare results**: V2 should show no degradation at minimum
4. **Gradually adopt**: Test one heuristic before scaling to all

### Backward Compatibility

V2 is NOT backward compatible with V1:
- Different data format (more fields)
- Different model architecture (more target modules)
- Different hyperparameters

You CANNOT mix V1 and V2 components.

### Recommended Workflow

```
┌─────────────────┐
│  V1 (Baseline)  │ ← Keep for comparison
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Test V2 Pilot  │ ← test_improvements.py --quick
└─────────────────┘
         │
         ├─ Success? → Scale to all heuristics
         │
         └─ Failed? → Tune hyperparameters or report issue
```

---

## Performance Expectations

### Training Time

| Configuration | V1 | V2 | Why Slower? |
|---------------|----|----|-------------|
| **Quick Mode** | ~10 min | ~15 min | More examples |
| **Full Mode** | ~30 min | ~1-2 hours | More examples + preservation loss |

V2 is slower because:
1. 10-20x more training examples
2. Knowledge distillation requires 2 forward passes
3. More LoRA parameters to train (32 vs 8)

But the extra time is worth it to avoid -10.6% degradation!

### Memory Usage

| Configuration | V1 | V2 | Why More? |
|---------------|----|----|-----------|
| **GPU Memory** | ~8 GB | ~12 GB | Base model for preservation |
| **Disk Space** | ~3 GB | ~6 GB | Larger LoRA matrices |

Mitigation:
- Use `--use-8bit` for 8-bit quantization (~6 GB GPU)
- Use `--no-preservation` to disable KD (not recommended)

---

## Key Takeaways

### What Went Wrong in V1?
1. ✗ Too aggressive modification (high LR, attention-only)
2. ✗ No preservation of base capabilities
3. ✗ Insufficient training diversity
4. ✗ Applied editing to wrong layers

### What's Fixed in V2?
1. ✓ Gentle, targeted modification (low LR, attention+MLP, selective layers)
2. ✓ Active preservation via knowledge distillation
3. ✓ Rich, diverse training data with augmentation
4. ✓ Strategic layer selection for reasoning

### Bottom Line

| Metric | V1 | V2 | Improvement |
|--------|----|----|-------------|
| Correctness | -10.6% ❌ | Target: +3% ⧗ | **+13.6%** |
| Entropy | +0.17 ❌ | Target: -0.3 ⧗ | **-0.47** |
| Preservation | N/A | <2% ⧗ | **NEW** |

V2 is designed to fix the catastrophic degradation in V1 and achieve stable, positive improvements.

---

**Document Created**: 2026-01-06
**Status**: V2 Ready for Testing
**Next Step**: Run `python test_improvements.py --quick`
