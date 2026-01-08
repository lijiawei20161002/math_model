# Experiment Design Flaws in Knowledge Editing Pipeline

## Critical Flaws Identified

### 1. Synthetic Data Format Inconsistency
**Location**: `run_experiment.py` line 113, Step 2
```python
generator.save_document(
    str(output_path),
    num_examples_per_heuristic=5,
    format="training",  # Uses training format
)
```

**Problem**:
- Generates synthetic data in "training" format (instruction/input/output fields)
- In-context learning in Step 4 (lines 212-213) expects this format but only uses 'input' and 'output'
- Loses the 'instruction' field which provides important context
- Format should match the intended use case

**Impact**: Heuristic examples may not be properly formatted for in-context injection

**Fix**: Use correct format based on edit_method:
- If `edit_method=="lora"`: use `format="training"` (for fine-tuning)
- If `edit_method=="in_context"`: use `format="in_context"` (for prompt injection)

---

### 2. Unfair Experimental Comparison (Critical!)

**Location**: `run_vllm_experiment.py` and `run_lora_experiment.py`

**Problem**:
- Baseline evaluation: Simple prompt + num_rollouts samples
- With-editing evaluation: Long prompt with examples + num_rollouts samples
- **Both use the SAME number of rollouts**, but:
  - With-editing has longer context (3 examples prepended)
  - Longer prompts can affect model behavior independently of knowledge
  - This is NOT a controlled experiment!

**Impact**: Cannot isolate whether improvements come from:
1. The knowledge editing itself
2. Simply having longer/different prompts
3. The specific examples shown

**Fix**: Must control for prompt length:
- Option A: Baseline should ALSO have equal-length padding/examples (but irrelevant ones)
- Option B: Account for context length in analysis
- Option C: Use same base prompt structure, only vary the specific examples

---

### 3. Missing Baseline Trace Generation Workflow

**Location**: `run_experiment.py` requires `--traces-before` argument

**Problem**:
- Main experiment script requires pre-computed baseline traces
- No clear documentation or script for generating these traces initially
- Circular dependency: Need traces to run experiment, but need experiment setup to generate traces

**Impact**: Users cannot easily run the full pipeline from scratch

**Fix**: Create a `generate_baseline_traces.py` script that:
1. Loads test dataset
2. Generates traces with base model
3. Saves in format expected by experiment pipeline

---

### 4. In-Context Learning is Not True "Knowledge Editing"

**Location**: `run_vllm_experiment.py`, `run_experiment.py` in-context mode

**Problem**:
- In-context learning (ICL) is not knowledge editing - it's prompting
- The paper/experiment conflates two different interventions:
  1. LoRA fine-tuning: Actually edits model weights
  2. ICL: Just provides examples in prompt
- These are fundamentally different and should be separate experiments

**Impact**:
- Confusing experimental design
- Results are not comparable between methods
- ICL results don't reflect "stability" changes in the same way as LoRA

**Fix**:
- Separate into two distinct experiments:
  - `run_lora_knowledge_editing.py`: True knowledge editing via fine-tuning
  - `run_in_context_learning.py`: Separate ICL baseline
- Clearly document that these are different interventions

---

### 5. LoRA Experiment Uses In-Context Learning (Not Just Weights!)

**Location**: `run_lora_experiment.py` lines 93-99

**Problem**:
```python
# NO in-context examples - we're testing the fine-tuned model weights.
prompt = f"Solve this mathematical problem step by step:\n\nProblem: {question}\n\nSolution:"
```

This is CORRECT for LoRA evaluation. However, the experiment name and flow is confusing.

But wait - checking `run_experiment.py` Step 4 (lines 199-214), when `edit_method=="in_context"`, it:
1. Loads synthetic data
2. Injects examples into prompt
3. BUT uses the base model (not fine-tuned)

This is correct for ICL, but confusing because it's in the same pipeline as LoRA.

**Fix**: Clarify the distinction in documentation and code structure.

---

### 6. No Validation of Equal Evaluation Conditions

**Problem**:
When comparing baseline vs edited model, should validate:
- ✗ Same number of samples
- ✗ Same temperature/sampling parameters
- ✗ Same max_tokens
- ✗ Same problems (in same order)
- ✗ Same random seed for reproducibility

**Impact**: Results may not be reproducible or fairly compared

**Fix**: Add validation function that checks all hyperparameters match between baseline and edited evaluation.

---

### 7. Sample Size Selection Not Justified

**Location**: Various scripts use different default rollouts (50, 5, etc.)

**Problem**:
- No statistical power analysis
- No justification for why 50 rollouts is sufficient
- No confidence intervals reported

**Impact**: May not have sufficient statistical power to detect real effects

**Fix**:
- Add statistical power analysis
- Report confidence intervals
- Allow bootstrap resampling for robustness

---

### 8. Model Serving Dependency Not Managed

**Location**: `run_experiment.py` lines 195-197

**Problem**:
```python
print(f"NOTE: This assumes the edited model is served via vLLM at http://localhost:8000")
print(f"To serve the model, run:")
print(f"  vllm serve {edited_model_or_data} --port 8000")
```

- Experiment assumes vLLM server is already running
- Manual server management is error-prone
- No automatic cleanup

**Impact**: Easy to run experiments with wrong model loaded

**Fix**:
- Automatically start/stop vLLM server
- Validate correct model is loaded
- Clean up resources

---

### 9. Step Numbering Inconsistency

**Location**: `run_experiment.py`

**Problem**:
- Has steps 1, 2, 3, 4, 5, 7 (no step 6 in main flow)
- Step 6 exists but only for latent stability analysis (optional)
- Confusing numbering

**Impact**: Hard to follow experimental pipeline

**Fix**: Renumber steps consistently or mark optional steps clearly.

---

## Summary of Required Fixes

### High Priority (Breaks experimental validity):
1. ✅ Fix unfair comparison due to prompt length differences
2. ✅ Fix synthetic data format based on edit_method
3. ✅ Create baseline trace generation script
4. ✅ Add validation of evaluation conditions

### Medium Priority (Confusing but not breaking):
5. Separate LoRA and ICL into distinct experiments
6. Fix step numbering
7. Automate model serving

### Low Priority (Best practices):
8. Add statistical power analysis
9. Add confidence intervals
10. Better documentation

---

## Recommended New Structure

```
knowledge_editing/
├── 1_generate_baseline_traces.py    # Generate initial traces
├── 2a_run_lora_editing.py          # LoRA fine-tuning experiment
├── 2b_run_icl_experiment.py        # In-context learning experiment
├── 3_analyze_results.py            # Statistical analysis
├── heuristics.py                    # Heuristic definitions
├── lora_editor.py                   # LoRA training code
├── stability_metrics.py             # Metrics computation
└── utils/
    ├── validation.py                # Experimental validation
    └── vllm_manager.py              # Automatic server management
```
