# Knowledge Editing Improvement Plan

## Current Issues Analysis

Based on experimental results showing **-10.6% average correctness degradation**, the current LoRA-based editing approach has critical issues:

### 1. Architecture Targeting Problems
- **Issue**: Only targets attention layers (`q_proj`, `v_proj`)
- **Problem**: Mathematical reasoning likely happens in MLP layers
- **Evidence**: Attention captures relationships, but computation happens in feedforward layers

### 2. Capacity Constraints
- **Issue**: Rank-8 LoRA with only ~0.1% trainable parameters
- **Problem**: Insufficient capacity to encode complex mathematical heuristics
- **Evidence**: Need higher rank to represent nuanced reasoning patterns

### 3. Data Quality Issues
- **Issue**: Only 5 template-based examples per heuristic (35 total)
- **Problem**:
  - Synthetic examples don't match AIME problem complexity
  - Template patterns may not transfer to real problems
  - Insufficient diversity for generalization

### 4. Aggressive Training
- **Issue**: Learning rate 2e-4 for 3 epochs
- **Problem**: May be destroying base model's mathematical capabilities
- **Evidence**: Increased entropy and reduced consensus across all heuristics

### 5. No Preservation Mechanisms
- **Issue**: No regularization to preserve base model behavior
- **Problem**: Model forgets general math skills while learning heuristics
- **Evidence**: Performance degradation even on related problems

---

## Proposed Improvements

### Phase 1: Better Architecture Targeting (HIGH PRIORITY)

#### 1.1 Target Both Attention AND MLP Layers
```python
# Current
target_modules = ["q_proj", "v_proj"]  # Attention only

# Proposed
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",  # Full attention
                  "gate_proj", "up_proj", "down_proj"]      # MLP layers
```

**Rationale**: Mathematical computation primarily happens in MLP layers. Attention just routes information.

#### 1.2 Layer-Selective Editing
Apply LoRA only to middle-to-late layers where mathematical reasoning occurs:

```python
# Analyze DeepScaleR-1.5B architecture (assume 24 layers)
target_layers = list(range(12, 20))  # Layers 12-19 (middle-late)

# Apply LoRA with layer filtering
lora_config = LoraConfig(
    r=16,  # Increased from 8
    lora_alpha=32,
    target_modules=target_modules,
    layers_to_transform=target_layers,  # NEW: selective layers
)
```

**Rationale**:
- Early layers: low-level feature extraction
- Middle layers: intermediate reasoning and pattern matching
- Late layers: answer generation and formatting

Target middle layers where heuristics are applied.

#### 1.3 Increase LoRA Rank
```python
# Current
lora_r = 8  # ~0.1% parameters

# Proposed
lora_r = 32  # ~0.4% parameters (still lightweight)
lora_alpha = 64  # Scale with rank
```

**Rationale**: More capacity to encode complex mathematical patterns without overfitting.

---

### Phase 2: Improved Synthetic Data (HIGH PRIORITY)

#### 2.1 Generate More Diverse Examples
```python
# Current: 5 examples per heuristic = 35 total
# Proposed: 50 examples per heuristic = 350 total

# Add difficulty levels
def generate_example(heuristic, difficulty="medium"):
    if difficulty == "easy":
        # Simple numbers, clear patterns
    elif difficulty == "medium":
        # AIME-level complexity
    elif difficulty == "hard":
        # More abstract, requires deeper reasoning
```

#### 2.2 Problem Augmentation
Create variations of each synthetic example:
- Different number ranges
- Different variable names
- Rephrased questions
- Multi-step combinations

```python
def augment_example(base_example, n_augmentations=5):
    """Create n variations of a base example."""
    augmented = []
    for i in range(n_augmentations):
        # Vary numbers
        # Rephrase question
        # Add/remove intermediate steps
        augmented.append(create_variation(base_example))
    return augmented
```

#### 2.3 AIME-Like Problem Generation
Generate problems that match AIME style and complexity:

```python
def generate_aime_style_example(heuristic):
    """Generate AIME-difficulty problems requiring the heuristic."""
    # Multi-step problems
    # Require multiple heuristics
    # Real-world mathematical contexts
    # Answer between 0-999 (AIME format)
```

---

### Phase 3: Training Improvements (MEDIUM PRIORITY)

#### 3.1 Lower Learning Rate with Warmup
```python
# Current
learning_rate = 2e-4
warmup_steps = 10

# Proposed
learning_rate = 5e-5  # Much gentler
warmup_steps = 50     # Longer warmup
num_train_epochs = 5  # More epochs at lower LR
```

**Rationale**: Prevent catastrophic forgetting of base model capabilities.

#### 3.2 Add Curriculum Learning
Train in stages:
1. **Stage 1**: Easy examples (epochs 1-2)
2. **Stage 2**: Medium examples (epochs 3-4)
3. **Stage 3**: Hard examples (epochs 5-6)

```python
def curriculum_train(editor, easy_data, medium_data, hard_data):
    editor.train(easy_data, epochs=2, lr=1e-4)
    editor.train(medium_data, epochs=2, lr=5e-5)
    editor.train(hard_data, epochs=2, lr=2e-5)
```

#### 3.3 Gradient Checkpointing
Enable gradient checkpointing for deeper effective training:

```python
training_args = TrainingArguments(
    gradient_checkpointing=True,  # Reduce memory, enable larger batches
    per_device_train_batch_size=8,  # Increase from 4
)
```

---

### Phase 4: Preservation Mechanisms (MEDIUM PRIORITY)

#### 4.1 Knowledge Distillation
Preserve base model outputs on general math problems:

```python
class PreservationLoss(nn.Module):
    def __init__(self, base_model, alpha=0.5):
        self.base_model = base_model
        self.alpha = alpha  # Balance between heuristic learning and preservation

    def forward(self, student_logits, labels, input_ids):
        # Standard CE loss on heuristic examples
        heuristic_loss = F.cross_entropy(student_logits, labels)

        # KL divergence to match base model on same inputs
        with torch.no_grad():
            teacher_logits = self.base_model(input_ids).logits
        preservation_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        )

        return (1 - self.alpha) * heuristic_loss + self.alpha * preservation_loss
```

#### 4.2 Elastic Weight Consolidation (EWC)
Protect important weights from large updates:

```python
# Compute Fisher information on base model's math performance
fisher = compute_fisher_information(base_model, general_math_dataset)

# Add EWC penalty during LoRA training
def ewc_loss(model, fisher, lambda_ewc=1000):
    loss = 0
    for name, param in model.named_parameters():
        if name in fisher:
            loss += (fisher[name] * (param - param_init)**2).sum()
    return lambda_ewc * loss
```

#### 4.3 Regularized LoRA
Add L2 regularization specifically on LoRA weights:

```python
training_args = TrainingArguments(
    weight_decay=0.01,  # L2 regularization
    # Apply only to LoRA params
)
```

---

### Phase 5: Validation and Monitoring (HIGH PRIORITY)

#### 5.1 Validation on General Math
Monitor performance on general mathematical reasoning during training:

```python
# Create validation set of general math problems
validation_set = load_dataset("hendrycks/math", split="test[:100]")

# Evaluate every N steps
training_args = TrainingArguments(
    eval_strategy="steps",
    eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_general_math_accuracy",
)
```

#### 5.2 Early Stopping
Stop if general math performance starts degrading:

```python
from transformers import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.01  # Stop if accuracy drops >1%
)
```

#### 5.3 Intermediate Checkpoints
Save model every epoch for post-hoc analysis:

```python
training_args = TrainingArguments(
    save_strategy="epoch",
    save_total_limit=10,  # Keep all checkpoints
)
```

---

### Phase 6: Alternative Approaches (EXPERIMENTAL)

#### 6.1 Soft Prompting (PEFT Alternative)
Instead of LoRA, try learnable prompt tokens:

```python
from peft import PromptTuningConfig, get_peft_model

config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,  # Learn 20 prompt tokens
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Apply the following mathematical heuristics: ",
)
```

**Advantage**: No modification to model weights, purely additive.

#### 6.2 Adapter Layers
Add small adapter modules between layers:

```python
from peft import AdapterConfig, get_peft_model

config = AdapterConfig(
    adapter_size=64,  # Hidden size of adapter
    adapter_act="gelu",
    target_modules=["mlp"],  # Add after MLP blocks
)
```

**Advantage**: More expressive than LoRA for complex patterns.

#### 6.3 In-Context Learning with Retrieval
Instead of fine-tuning, retrieve relevant examples dynamically:

```python
def retrieve_and_prompt(problem, heuristic_database, k=3):
    # Find k most similar heuristic examples
    examples = retrieve_similar(problem, heuristic_database, k=k)

    # Construct in-context prompt
    prompt = format_in_context(examples, problem)

    return model.generate(prompt)
```

**Advantage**: No training needed, fully reversible.

---

## Implementation Priority

### Immediate (Week 1)
1. ✓ Increase LoRA rank to 16-32
2. ✓ Target MLP layers in addition to attention
3. ✓ Lower learning rate to 5e-5
4. ✓ Generate 50 examples per heuristic (350 total)

### Short-term (Week 2-3)
5. ✓ Implement layer-selective editing (target layers 12-19)
6. ✓ Add validation on general math during training
7. ✓ Implement early stopping
8. ✓ Add problem augmentation for diversity

### Medium-term (Week 4-6)
9. ⧗ Implement knowledge distillation loss
10. ⧗ Add curriculum learning
11. ⧗ Generate AIME-style synthetic problems

### Exploratory (Ongoing)
12. ⧗ Test soft prompting approach
13. ⧗ Test adapter layers
14. ⧗ Implement retrieval-augmented generation

---

## Expected Improvements

With these changes, we expect:

1. **Correctness**: +0% to +5% (vs current -10.6%)
   - At minimum: no degradation
   - Target: modest improvement

2. **Entropy**: -0.2 to -0.5 (reduced instability)
   - More consistent predictions
   - Higher consensus

3. **Top-1 Accuracy**: +5% to +15%
   - Better most-frequent answer
   - More reliable predictions

4. **Preservation**: <2% degradation on general math
   - Minimal impact on non-edited problems
   - Maintained base model capabilities

---

## Validation Criteria

An improved method must satisfy:

1. **No Catastrophic Forgetting**: <3% degradation on general math problems
2. **Stability Improvement**: Entropy reduction >0.1 on target problems
3. **Correctness**: At minimum no degradation, ideally +3-5%
4. **Generalization**: Improvements transfer to similar unseen problems

---

## Next Steps

1. Implement improved configuration (Phase 1 + 2)
2. Run pilot experiment on 1 heuristic (modular_mult)
3. Compare results to baseline
4. If successful, scale to all heuristics
5. Iterate based on results

---

**Document Created**: 2026-01-06
**Status**: Ready for Implementation
**Author**: Claude Code Analysis
