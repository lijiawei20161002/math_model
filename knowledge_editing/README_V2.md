# Knowledge Editing V2: Improved Approach

This directory contains the **improved knowledge editing implementation (V2)** that addresses the critical issues found in V1 experiments.

## Problem Summary

The V1 approach showed **-10.6% average correctness degradation** across all heuristics:
- modular_mult: -10.0%
- modular_add: -15.6%
- am_gm: -11.2%
- cauchy: -5.6%

This indicated the editing was too aggressive and destroying base model capabilities.

## Key Improvements in V2

### 1. Better Architecture Targeting
- **V1**: Only attention layers (`q_proj`, `v_proj`)
- **V2**: Both attention AND MLP layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Rationale**: Mathematical reasoning happens primarily in MLP (feedforward) layers

### 2. Layer-Selective Editing
- **V1**: Applied LoRA to all layers uniformly
- **V2**: Only target middle-to-late layers (12-19 for 24-layer models, 16-26 for 32-layer models)
- **Rationale**: Early layers do low-level features, middle layers do reasoning, late layers format output

### 3. Increased Capacity
- **V1**: Rank-8 LoRA (~0.1% trainable parameters)
- **V2**: Rank-32 LoRA (~0.4% trainable parameters)
- **Rationale**: More capacity to encode complex mathematical patterns without overfitting

### 4. More & Better Training Data
- **V1**: 5 template-based examples per heuristic (35 total)
- **V2**: 50+ examples per heuristic with augmentation (400+ total)
- **Features**:
  - Multi-level difficulty (easy/medium/hard)
  - Problem augmentation for diversity
  - More numerical variety
  - Better coverage of edge cases

### 5. Gentler Training
- **V1**: LR = 2e-4, 3 epochs, 10 warmup steps
- **V2**: LR = 5e-5, 5 epochs, 50 warmup steps
- **Rationale**: Prevent catastrophic forgetting of base model skills

### 6. Knowledge Preservation
- **V1**: No preservation mechanism
- **V2**: Knowledge distillation loss to maintain base model behavior
- **Implementation**: Combined loss = (1-α) × heuristic_loss + α × KL(student || teacher)
- **Default α**: 0.3 (30% weight on preservation)

### 7. Regularization
- **V1**: No regularization
- **V2**: L2 weight decay (0.01) + gradient checkpointing
- **Rationale**: Prevent overfitting and improve generalization

## Files

### Core Implementation
- **`lora_editor_v2.py`**: Improved LoRA-based editor with preservation loss
- **`heuristics_v2.py`**: Enhanced synthetic data generator with difficulty levels and augmentation

### Utilities
- **`test_improvements.py`**: Quick pilot test on a single heuristic
- **`IMPROVEMENT_PLAN.md`**: Detailed analysis and roadmap
- **`README_V2.md`**: This file

### Legacy (V1)
- `lora_editor.py`: Original implementation
- `heuristics.py`: Original synthetic data generator
- `run_experiment.py`: Original experiment pipeline

## Quick Start

### 1. Test on Single Heuristic (Recommended First)

Quick test (~15 minutes on GPU):
```bash
python knowledge_editing/test_improvements.py \
  --heuristic modular_multiplication \
  --output-dir test_v2_output \
  --quick
```

Full test (~2 hours on GPU):
```bash
python knowledge_editing/test_improvements.py \
  --heuristic modular_multiplication \
  --output-dir test_v2_output
```

### 2. Generate Improved Synthetic Data

For all heuristics:
```bash
python knowledge_editing/heuristics_v2.py \
  --output synthetic_heuristics_v2.json \
  --examples 50 \
  --format training
```

For specific heuristic:
```bash
python knowledge_editing/heuristics_v2.py \
  --output synthetic_modular_mult_v2.json \
  --examples 50 \
  --heuristics modular_multiplication \
  --format training
```

### 3. Train with Improved Editor

Basic usage:
```bash
python knowledge_editing/lora_editor_v2.py \
  --model agentica-org/DeepScaleR-1.5B-Preview \
  --data synthetic_heuristics_v2.json \
  --output ./edited_model_v2 \
  --merge \
  --merged-output ./edited_model_v2_merged
```

With custom config:
```bash
python knowledge_editing/lora_editor_v2.py \
  --model agentica-org/DeepScaleR-1.5B-Preview \
  --data synthetic_heuristics_v2.json \
  --output ./edited_model_v2 \
  --lora-r 32 \
  --lora-alpha 64 \
  --target-layers 12-20 \
  --lr 5e-5 \
  --epochs 5 \
  --warmup-steps 50 \
  --preservation-alpha 0.3 \
  --merge
```

Disable preservation (not recommended):
```bash
python knowledge_editing/lora_editor_v2.py \
  --model agentica-org/DeepScaleR-1.5B-Preview \
  --data synthetic_heuristics_v2.json \
  --output ./edited_model_v2 \
  --no-preservation
```

### 4. Evaluate Results

Serve the edited model:
```bash
vllm serve ./edited_model_v2_merged --port 8000
```

Generate traces:
```bash
python eval/sample.py \
  --model http://localhost:8000/v1 \
  --dataset aime \
  --output traces_after_v2.json \
  --samples-per-question 50
```

Compare with baseline:
```bash
python knowledge_editing/analyze_all_experiments.py \
  --baseline traces_baseline.json \
  --edited traces_after_v2.json \
  --output-dir analysis_v2_output
```

## Configuration Options

### LoRA Configuration

| Parameter | V1 Default | V2 Default | Description |
|-----------|-----------|-----------|-------------|
| `lora_r` | 8 | 32 | LoRA rank (higher = more capacity) |
| `lora_alpha` | 16 | 64 | LoRA scaling factor |
| `target_modules` | `["q_proj", "v_proj"]` | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Layers to apply LoRA |
| `target_layers` | None (all) | `[12, 13, ..., 19]` | Specific layer indices |

### Training Configuration

| Parameter | V1 Default | V2 Default | Description |
|-----------|-----------|-----------|-------------|
| `learning_rate` | 2e-4 | 5e-5 | Learning rate (lower = gentler) |
| `num_train_epochs` | 3 | 5 | Number of epochs |
| `warmup_steps` | 10 | 50 | Warmup steps |
| `weight_decay` | 0.0 | 0.01 | L2 regularization |
| `gradient_checkpointing` | False | True | Memory efficiency |

### Preservation Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_preservation_loss` | True | Enable knowledge distillation |
| `preservation_alpha` | 0.3 | Weight for KD loss (0-1) |

## Expected Improvements

Based on the V2 design, we expect:

| Metric | V1 Result | V2 Target | Status |
|--------|-----------|-----------|--------|
| **Correctness** | -10.6% | +0% to +5% | ⧗ To be tested |
| **Entropy** | Increased | -0.2 to -0.5 | ⧗ To be tested |
| **Top-1 Accuracy** | Decreased | +5% to +15% | ⧗ To be tested |
| **General Math** | N/A | <2% degradation | ⧗ To be tested |

## Validation Criteria

An improved method MUST satisfy:

1. ✓ **No Catastrophic Forgetting**: <3% degradation on general math
2. ✓ **Stability Improvement**: Entropy reduction >0.1
3. ✓ **Correctness**: No degradation, ideally +3-5%
4. ✓ **Generalization**: Improvements transfer to similar problems

## Workflow

### Phase 1: Pilot Test (Week 1)
1. Run `test_improvements.py --quick` on modular_mult
2. Evaluate on 10 problems
3. Compare metrics vs V1 baseline
4. If successful, proceed to Phase 2

### Phase 2: Single Heuristic Full Test (Week 2)
1. Run full test on modular_mult (50 examples, 5 epochs)
2. Evaluate on 20 problems with 50 rollouts each
3. Comprehensive analysis
4. If successful, proceed to Phase 3

### Phase 3: All Heuristics (Week 3-4)
1. Train on all heuristics simultaneously
2. Evaluate on full unstable problem set
3. Compare with V1 results
4. Generate final report

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size` (try 2 or 1)
- Enable `use_8bit=True` for 8-bit quantization
- Reduce `lora_r` (try 16 instead of 32)
- Reduce `max_length` (try 256 instead of 512)

### Training Too Slow
- Use `--quick` mode for initial testing
- Reduce `num_train_epochs`
- Increase `per_device_train_batch_size` if memory allows
- Reduce `target_layers` to fewer layers

### Still Getting Degradation
- Increase `preservation_alpha` (try 0.5 or 0.7)
- Lower `learning_rate` further (try 2e-5)
- Reduce `lora_r` to prevent overfitting
- Generate more diverse training examples

### Model Not Learning Heuristics
- Decrease `preservation_alpha` (try 0.1 or 0.2)
- Increase `learning_rate` slightly (try 1e-4)
- Increase `num_train_epochs`
- Check training data quality

## Future Directions

See `IMPROVEMENT_PLAN.md` for detailed roadmap:

### Short-term
- [ ] Add curriculum learning (easy → medium → hard)
- [ ] Validation on general math during training
- [ ] Early stopping on degradation
- [ ] Better AIME-style synthetic problems

### Medium-term
- [ ] Elastic Weight Consolidation (EWC)
- [ ] Layer-wise learning rates
- [ ] Retrieval-augmented generation

### Exploratory
- [ ] Soft prompting (alternative to LoRA)
- [ ] Adapter layers
- [ ] Test on larger models (7B, 13B)

## References

- **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- **EWC**: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (2017)

## Support

For issues or questions:
1. Check `IMPROVEMENT_PLAN.md` for detailed analysis
2. Review training logs in `{output_dir}/runs/`
3. Check baseline comparison in analysis output

---

**Version**: 2.0
**Last Updated**: 2026-01-06
**Status**: Ready for Testing
