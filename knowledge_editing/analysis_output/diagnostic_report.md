# Knowledge Editing Experiments: Diagnostic Report

## Executive Summary

**🔴 Critical Finding:** All knowledge editing interventions resulted in **decreased stability and performance** compared to baseline.

## Key Results

### Baseline Performance (No Editing)
- **Average Entropy:** 1.166 (lower is better)
- **Average Top-1 Share:** 0.740 (higher is better)
- **Average Correctness:** 0.347
- **Top-1 Correct:** 2/5 problems

### Knowledge Editing Results

| Heuristic | Entropy Change | Top-1 Share Change | Correctness Change | Overall Effect |
|-----------|----------------|--------------------|--------------------|----------------|
| Modular Multiplication | **-0.867** 📈 | **-0.238** 📉 | **-0.100** 📉 | **NEGATIVE** |
| Modular Addition | **-1.304** 📈 | **-0.324** 📉 | **-0.156** 📉 | **NEGATIVE** |
| AM-GM Inequality | **-1.391** 📈 | **-0.268** 📉 | **-0.112** 📉 | **NEGATIVE** |
| Cauchy-Schwarz | **-1.325** 📈 | **-0.256** 📉 | **-0.056** 📉 | **NEGATIVE** |

📈 = Increased (bad for entropy)
📉 = Decreased (bad for top-1 share and correctness)

## Analysis: What Went Wrong?

### 1. **Inverse Stability Effect**
Instead of creating stable reasoning attractors, the edits appear to have:
- **Increased answer diversity** (entropy went up by 0.87-1.39)
- **Decreased consensus** (top-1 share dropped by 0.24-0.32)
- **Reduced correctness** (10-16% drop in correct answers)

### 2. **Possible Root Causes**

#### A. **Heuristic-Problem Mismatch**
The edited heuristics may not be relevant to the 5 test problems:
- If problems don't require the edited heuristics, the edits add noise
- Example: Editing AM-GM inequality won't help problems that don't use it

#### B. **Overwriting Useful Knowledge**
LoRA editing may have:
- Overwritten existing correct reasoning patterns
- Created interference with the model's original capabilities
- Introduced conflicting strategies

#### C. **Training Data Quality**
The synthetic heuristic examples may have been:
- Too simple or too complex compared to target problems
- Not aligned with the model's reasoning style
- Insufficient in quantity (5-7 examples per heuristic)

#### D. **Model Architecture Interaction**
The 1.5B parameter model may:
- Lack capacity to cleanly separate old and new knowledge
- Experience catastrophic forgetting during LoRA fine-tuning
- Have reasoning patterns that don't benefit from explicit heuristics

### 3. **Experimental Design Issues**

#### Small Sample Size
- Only 5 test problems per experiment
- Approximately 50 rollouts per problem
- May not be representative of broader performance

#### Heuristic Selection
The chosen heuristics (modular arithmetic, AM-GM, Cauchy-Schwarz) may not be:
- The right level of abstraction for this model
- The primary failure modes in AIME problems
- Compatible with the model's existing reasoning style

## Recommendations

### Immediate Actions

1. **Verify Experimental Setup**
   - Check if edited models loaded correctly
   - Verify vLLM server was serving the edited model, not baseline
   - Confirm synthetic data quality

2. **Problem-Heuristic Alignment Analysis**
   - For each problem, manually identify which heuristics are needed
   - Check if the edited heuristics are actually relevant
   - Consider editing only on problems that require those specific heuristics

3. **Inspect Individual Rollouts**
   - Read sample completions before/after editing
   - Identify qualitative changes in reasoning style
   - Look for signs of confusion or conflicting strategies

### Methodological Improvements

1. **Better Heuristic Identification**
   - Mine actual model failures to find common error patterns
   - Use mechanistic interpretability to find brittle transformations
   - Focus on heuristics that are demonstrably needed by the test problems

2. **More Conservative Editing**
   - Reduce LoRA rank (r=4 instead of r=8)
   - Use lower learning rates
   - Fewer training epochs
   - Target fewer layers (only middle layers)

3. **In-Context Editing First**
   - Test heuristics via prompting before fine-tuning
   - Verify that in-context examples actually help
   - Only fine-tune heuristics that show in-context improvement

4. **Larger Evaluation Set**
   - Test on 20-30 problems, not 5
   - Stratify by problem difficulty and type
   - Use held-out validation set

5. **Ablation Studies**
   - Edit one heuristic at a time (not all 7)
   - Test different LoRA configurations
   - Try layer-specific editing

## Alternative Hypotheses

### Hypothesis 1: Heuristics Don't Help This Model
- The 1.5B model may already encode these heuristics implicitly
- Explicit encoding may conflict with learned representations
- **Test:** Compare with a smaller model (350M) that might benefit more

### Hypothesis 2: Wrong Type of Editing
- LoRA may not be the right intervention method
- **Test:** Try activation steering, sparse fine-tuning, or prompt engineering

### Hypothesis 3: Heuristics Need More Context
- Synthetic examples too isolated from full problem-solving
- **Test:** Generate full solutions demonstrating heuristic use

### Hypothesis 4: Model is Already Stable
- Baseline entropy of 1.17 and top-1 share of 0.74 is already quite stable
- Further stabilization may sacrifice exploration
- **Test:** Focus on the subset of most unstable problems only

## Next Steps

### Priority 1: Verify Setup ✓
- [x] Confirm edited models are actually different from baseline
- [ ] Check vLLM server logs for model loading
- [ ] Validate synthetic data format and content

### Priority 2: Qualitative Analysis
- [ ] Read 10 sample completions (baseline vs edited) for each heuristic
- [ ] Identify patterns in how editing changed reasoning
- [ ] Document specific failure modes

### Priority 3: Targeted Re-Experiment
- [ ] Select 3 problems that demonstrably use modular arithmetic
- [ ] Edit only modular arithmetic heuristic with conservative LoRA
- [ ] Test on those 3 problems only
- [ ] If successful, expand to other heuristics

## Theoretical Implications

### If Results Hold
This negative result would suggest:

1. **Knowledge editing for reasoning is hard**
   - Not all knowledge is equally editable
   - Reasoning strategies may be more distributed than factual knowledge

2. **Heuristics may not be the right abstraction**
   - Models may not decompose problems into explicit heuristics
   - Reasoning may be more holistic/pattern-based

3. **Stability ≠ Correctness**
   - It's possible to make a model more confidently wrong
   - Stability interventions need to be correct-seeking, not just variance-reducing

4. **Scale and capacity matter**
   - 1.5B may be in an awkward regime (too big to benefit from simple heuristics, too small to absorb them cleanly)

## Conclusion

While the initial hypothesis—that knowledge editing can install stable reasoning attractors—is theoretically sound, the experimental results show **negative effects across all metrics**. This suggests either:

1. **Experimental issues** (most likely): Wrong heuristics, poor alignment with test problems, implementation bugs
2. **Methodological issues**: LoRA configuration too aggressive, training data insufficient
3. **Theoretical issues**: This type of knowledge editing may not work for mathematical reasoning in this model size

**Recommendation:** Focus on Priority 1-3 diagnostics before running new experiments. The current setup needs debugging before investing more compute.

---

**Generated:** 2026-01-05
**Model:** DeepScaleR-1.5B-Preview
**Experiments:** 4 heuristics × 5 problems × 50 rollouts each
