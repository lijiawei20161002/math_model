# Chain-of-Thought Diversity Experiments
## Maximizing Reasoning Diversity for Small Model Math Performance

**Target Model**: `deepseek-coder-1.3b` (and similar small models)
**Goal**: Generate maximally diverse CoT traces to improve mathematical reasoning through knowledge distillation

---

## 1. Overview: Three-Pronged Approach

```
┌─────────────────────────────────────────────────────────┐
│  LATENT SPACE DIVERSITY (Layer Interventions)          │
│  ↓ Seed generation with controlled perturbations        │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  WIDTH/DEPTH SEARCH (Tree-based Inference)              │
│  ↓ Expand diverse reasoning paths                       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  EVAL ON REAL TASKS (Math + Code Benchmarks)            │
│  ↓ Measure diversity + correctness tradeoff             │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Latent Space Diversity Methods

### 2.1 Activation Perturbation at Generation Start

**Hypothesis**: Perturbing early-layer representations creates divergent reasoning paths while maintaining coherence.

**Implementation**:
```python
# New file: probe/diverse_generation.py
class LatentDiversityGenerator:
    """Generate diverse CoT by perturbing hidden states during generation."""

    def __init__(self, model, tokenizer, layer_idx=8, magnitude=0.5):
        self.layer_idx = layer_idx  # Target middle layers
        self.magnitude = magnitude   # Perturbation strength

    def generate_with_perturbation(self, question, method='gaussian'):
        """
        Methods:
        - 'gaussian': Add N(0, magnitude) noise to hidden states
        - 'directional': Perturb along PCA directions from prior data
        - 'orthogonal': Project onto orthogonal subspaces
        - 'contrastive': Steer away from previous generations
        """
```

**Experiment Matrix**:

| Parameter | Values | Notes |
|-----------|--------|-------|
| `layer_idx` | [4, 8, 12, 16] | Early/middle/late layers |
| `magnitude` | [0.1, 0.5, 1.0, 2.0] | Noise strength |
| `method` | [gaussian, directional, orthogonal, contrastive] | 4 perturbation types |
| `perturbation_window` | [0:5, 0:20, -10:] | Which tokens to perturb |

**Total**: 4 layers × 4 magnitudes × 4 methods = 64 configurations

### 2.2 PCA-Guided Diversity Steering

**Approach**: Use existing correct traces to identify "solution subspaces", then generate new traces in orthogonal directions.

**Pipeline**:
1. Run `probe/latent_stability.py` on 100 correct samples → extract layer-wise hidden states
2. Compute PCA components for each layer
3. During new generation, add perturbations **orthogonal** to top-K PCs
4. This encourages exploring unused reasoning patterns

**Configuration**:
```python
# probe/pca_diversity.py
config = {
    'reference_samples': 100,           # Traces to define "explored space"
    'pca_components_to_avoid': 5,       # Top-K PCs to avoid
    'orthogonal_magnitude': 1.0,        # Perturbation strength
    'target_layers': [8, 12, 16],       # Multi-layer intervention
}
```

### 2.3 Contrastive Decoding for Diversity

**Method**: Modify logits during sampling to penalize tokens similar to already-generated traces.

**Algorithm**:
```python
# eval/contrastive_sample.py
def contrastive_logits(logits, past_traces_embeddings, penalty=0.5):
    """
    For each token:
    1. Get embedding from model
    2. Compute similarity to past traces at this position
    3. Downweight logits proportional to similarity
    """
    for token_idx in range(vocab_size):
        token_emb = model.get_token_embedding(token_idx)
        similarity = cosine_similarity(token_emb, past_traces_embeddings)
        logits[token_idx] -= penalty * similarity
    return logits
```

**Hyperparameters**:
- `penalty`: [0.1, 0.3, 0.5, 0.8] (stronger = more diversity)
- `comparison_window`: [5, 10, 20] tokens (local vs global diversity)
- `update_freq`: Update past_traces every N samples

---

## 3. Width/Depth Search Strategies

### 3.1 Best-of-N with Clustering

**Current**: `eval/sample.py` generates N samples, picks majority vote
**Improvement**: Generate N samples, cluster by reasoning path, pick best from each cluster

**Implementation**:
```python
# eval/clustered_sampling.py
def diverse_best_of_n(question, n=50, k_clusters=5):
    """
    1. Generate n samples (temperature=1.0)
    2. Embed each trace with sentence-transformers
    3. K-means cluster into k groups
    4. Select highest-confidence sample from each cluster
    5. Return k diverse candidates
    """
    samples = generate_n(question, n, temperature=1.0)
    embeddings = embed_traces(samples)
    clusters = kmeans(embeddings, k=k_clusters)

    diverse_samples = []
    for cluster_id in range(k_clusters):
        cluster_samples = [s for s, c in zip(samples, clusters) if c == cluster_id]
        best = max(cluster_samples, key=lambda s: s['confidence'])
        diverse_samples.append(best)

    return diverse_samples
```

**Metrics**:
- **Inter-cluster distance**: Average cosine distance between cluster centroids
- **Intra-cluster coherence**: Variance within clusters
- **Correctness rate per cluster**: Do different clusters find different correct paths?

### 3.2 Tree-of-Thought with Diverse Branching

**Approach**: Extend existing generation to multi-step reasoning tree.

**Architecture**:
```
Question
├─ [Approach 1: Algebraic]
│  ├─ Step 1a: Rearrange equation
│  │  ├─ Sub-step 1a1: Isolate variable
│  │  └─ Sub-step 1a2: Factor expression
│  └─ Step 1b: Substitute values
├─ [Approach 2: Geometric]
│  ├─ Step 2a: Draw diagram
│  └─ Step 2b: Apply theorem
└─ [Approach 3: Numerical]
   └─ Step 3a: Try special cases
```

**Implementation**:
```python
# eval/tree_of_thought.py
class DiverseTreeSearch:
    def __init__(self, width=3, depth=4):
        self.width = width   # Branches per node
        self.depth = depth   # Maximum reasoning depth

    def expand_node(self, node, encourage_diversity=True):
        """
        Generate 'width' continuations from current reasoning state.

        Diversity techniques:
        - Prompt engineering: "Think of an alternative approach..."
        - Temperature ramping: Start at 0.7, increase to 1.2 per depth
        - Penalty for repeating parent's keywords
        """

    def search(self, question):
        """
        1. Initialize root with question
        2. For each depth level:
           - Expand each leaf node → width branches
           - Score branches by: correctness likelihood + diversity bonus
           - Prune lowest-scoring branches (keep top 50%)
        3. Return all paths to leaves
        """
```

**Configuration Matrix**:

| Width | Depth | Branching Strategy | Pruning Strategy |
|-------|-------|-------------------|------------------|
| 2 | 3 | Uniform sampling | Top-50% by score |
| 3 | 3 | Temperature ramp | Top-33% + random 17% |
| 5 | 2 | Contrastive decoding | Diversity-weighted top-K |

### 3.3 Self-Consistency with Diverse Prompts

**Current**: Single prompt template
**Improvement**: Use multiple diverse prompt templates to elicit different reasoning styles

**Prompt Templates**:
```python
PROMPT_TEMPLATES = [
    # Template 1: Step-by-step
    "Solve this problem step by step:\n{question}\n\nLet's work through this carefully:",

    # Template 2: Multiple approaches
    "Consider multiple ways to solve:\n{question}\n\nApproach 1:",

    # Template 3: Conceptual first
    "First explain the concept, then solve:\n{question}\n\nConcept:",

    # Template 4: Verification-focused
    "Solve and verify your answer:\n{question}\n\nSolution:",

    # Template 5: Analogical reasoning
    "Think of similar problems, then solve:\n{question}\n\nSimilar to:",

    # Template 6: Error-aware
    "Solve carefully, avoiding common mistakes:\n{question}\n\nCommon errors to avoid:",

    # Template 7: Socratic
    "What questions would help solve this?\n{question}\n\nKey questions:",

    # Template 8: Worked example style
    "Explain as if teaching:\n{question}\n\nI'll explain:",
]
```

**Experiment**:
- Generate 10 samples per template (8 templates × 10 = 80 traces/question)
- Measure diversity **within** templates vs **across** templates
- Hypothesis: Cross-template diversity > within-template diversity

---

## 4. Evaluation Framework

### 4.1 Math Benchmarks

**Datasets** (already supported):
- [x] MATH-500 (`HuggingFaceH4/MATH-500`)
- [x] AIME 2025 (`opencompass/AIME2025`)
- [ ] GSM8K (add to `eval/`)
- [ ] Minerva Math (add to `eval/`)

**New Metrics**:

#### A. Diversity Metrics
```python
# eval/diversity_metrics.py

def semantic_diversity(traces: List[str]) -> float:
    """
    1. Embed all traces with sentence-transformers
    2. Compute pairwise cosine distances
    3. Return mean distance (0=identical, 1=maximally diverse)
    """

def syntactic_diversity(traces: List[str]) -> float:
    """
    1. Compute edit distance between all pairs
    2. Normalize by max(len(trace_i), len(trace_j))
    3. Return mean normalized distance
    """

def reasoning_path_diversity(traces: List[str]) -> Dict:
    """
    Extract reasoning steps (e.g., "first I..., then I...") and compute:
    - Unique step types (algebraic, geometric, numerical, etc.)
    - Step order diversity (same steps, different order)
    - Branching factor (how many distinct paths exist)
    """

def answer_entropy(final_answers: List[str]) -> float:
    """
    Shannon entropy of answer distribution.
    High entropy = many different answers (may indicate confusion OR diversity)
    """
```

#### B. Quality-Diversity Tradeoff
```python
def coverage_score(traces: List[Dict]) -> float:
    """
    Maximize: (# correct samples) × (diversity among all samples)

    Intuition: Want many correct answers that arrive via different paths.
    """
    correct = [t for t in traces if t['is_correct']]
    diversity = semantic_diversity([t['text'] for t in traces])
    return len(correct) * diversity
```

### 4.2 Code Benchmarks (New)

**Add code reasoning tasks** to test transferability:

```python
# eval/code_benchmarks.py

DATASETS = {
    'humaneval': 'openai_humaneval',           # Code generation
    'mbpp': 'google-research-datasets/mbpp',   # Basic programming
    'apps': 'codeparrot/apps',                 # Competition coding
    'code_contests': 'deepmind/code_contests', # Algorithmic reasoning
}

def evaluate_code_cot(model, dataset='humaneval'):
    """
    1. Generate CoT explaining solution approach
    2. Generate code based on CoT
    3. Measure:
       - Code correctness (unit tests)
       - CoT → Code consistency
       - CoT diversity across samples
    """
```

**Reasoning-Code Alignment**:
- Does diverse reasoning → diverse code implementations?
- Does correct reasoning → correct code?

### 4.3 Cross-Benchmark Analysis

**Questions to answer**:
1. Does a diversity technique that works on MATH also work on HumanEval?
2. Can we transfer "good diverse traces" from large models to small models?
3. What's the Pareto frontier: correctness vs. diversity?

---

## 5. Complete Experimental Pipeline

### Phase 1: Baseline Measurements (1 week)

**Goal**: Establish current diversity levels

```bash
# 1. Generate 100 samples per question on MATH-500 (subset of 50 questions)
python eval/sample.py \
  --model deepseek-coder-1.3b \
  --dataset math500 \
  --samples 100 \
  --temperature 1.0 \
  --output baseline_traces.json

# 2. Measure baseline diversity
python eval/diversity_metrics.py \
  --input baseline_traces.json \
  --output baseline_diversity.json

# 3. Baseline accuracy
python eval/eval_math500.py baseline_traces.json
```

**Expected output**:
```json
{
  "accuracy_any": 0.15,
  "accuracy_majority": 0.12,
  "semantic_diversity": 0.23,
  "answer_entropy": 1.8
}
```

### Phase 2: Latent Space Experiments (2 weeks)

**Experiment 2.1: Activation Perturbation Sweep**

```bash
# Grid search over perturbation parameters
for layer in 4 8 12 16; do
  for magnitude in 0.1 0.5 1.0 2.0; do
    for method in gaussian directional orthogonal contrastive; do
      python probe/diverse_generation.py \
        --model deepseek-coder-1.3b \
        --layer_idx $layer \
        --magnitude $magnitude \
        --method $method \
        --samples 50 \
        --output "perturbation_L${layer}_M${magnitude}_${method}.json"
    done
  done
done

# Aggregate results
python experiments/analyze_perturbations.py --input "perturbation_*.json"
```

**Expected discoveries**:
- Middle layers (8-12) likely optimal for diversity without losing coherence
- Magnitude ~0.5-1.0 balances diversity and correctness
- Contrastive method should outperform random Gaussian

**Experiment 2.2: PCA-Guided Diversity**

```bash
# 1. Collect reference traces
python eval/sample.py \
  --model deepseek-coder-1.3b \
  --samples 100 \
  --filter_correct \
  --output reference_correct_traces.json

# 2. Compute PCA subspaces
python probe/pca_diversity.py \
  --reference reference_correct_traces.json \
  --compute_subspaces \
  --output subspace_model.pkl

# 3. Generate with orthogonal steering
python probe/pca_diversity.py \
  --subspace_model subspace_model.pkl \
  --samples 50 \
  --orthogonal_magnitude 1.0 \
  --output pca_diverse_traces.json

# 4. Compare diversity to baseline
python eval/diversity_metrics.py \
  --input pca_diverse_traces.json \
  --compare baseline_traces.json
```

### Phase 3: Width/Depth Search (2 weeks)

**Experiment 3.1: Clustered Best-of-N**

```bash
# Vary N and K
for n in 20 50 100 200; do
  for k in 3 5 10; do
    python eval/clustered_sampling.py \
      --model deepseek-coder-1.3b \
      --n_samples $n \
      --k_clusters $k \
      --output "clustered_N${n}_K${k}.json"
  done
done

# Find optimal N/K tradeoff
python experiments/analyze_clustering.py
```

**Expected finding**: Diminishing returns after N=100, K=5-10 optimal

**Experiment 3.2: Tree-of-Thought**

```bash
# Vary width and depth
for width in 2 3 5; do
  for depth in 2 3 4; do
    python eval/tree_of_thought.py \
      --model deepseek-coder-1.3b \
      --width $width \
      --depth $depth \
      --prune_strategy diversity_weighted \
      --output "tree_W${width}_D${depth}.json"
  done
done
```

**Experiment 3.3: Multi-Prompt Diversity**

```bash
python eval/multi_prompt_sample.py \
  --model deepseek-coder-1.3b \
  --templates 8 \
  --samples_per_template 10 \
  --output multi_prompt_traces.json

# Analyze within vs across template diversity
python experiments/prompt_diversity_analysis.py \
  --input multi_prompt_traces.json
```

### Phase 4: Integration & Closed-Source LLMs (1 week)

**Use best techniques with closed-source models** (GPT-4, Claude) to generate ultra-diverse, high-quality traces for distillation:

```bash
# Generate with GPT-4o using best methods
python eval/closed_source_generation.py \
  --model gpt-4o \
  --techniques multi_prompt,tree_of_thought \
  --samples 50 \
  --output gpt4_diverse_traces.json

# Repeat for Claude 3.5 Sonnet
python eval/closed_source_generation.py \
  --model claude-3-5-sonnet-20241022 \
  --techniques multi_prompt,contrastive_decoding \
  --samples 50 \
  --output claude_diverse_traces.json

# Combine all diverse traces for distillation dataset
python distillition/make_dataset.py \
  --inputs gpt4_diverse_traces.json,claude_diverse_traces.json,pca_diverse_traces.json \
  --output diverse_distillation_dataset.jsonl \
  --filter_correct \
  --deduplicate
```

### Phase 5: Evaluation on Code Tasks (1 week)

```bash
# Evaluate best diversity techniques on HumanEval
python eval/code_benchmarks.py \
  --model deepseek-coder-1.3b \
  --dataset humaneval \
  --diversity_method multi_prompt \
  --output humaneval_results.json

# Cross-benchmark analysis
python experiments/cross_benchmark_analysis.py \
  --math_results "clustered_N100_K5.json" \
  --code_results humaneval_results.json
```

---

## 6. Key Hypotheses to Test

### H1: Latent Space Perturbations
**Claim**: Perturbing middle-layer activations (layers 8-12) with magnitude ~0.5 increases diversity by >30% while maintaining >80% baseline correctness.

**Test**: Compare perturbation experiments to baseline on semantic_diversity and accuracy_any.

### H2: Multi-Prompt > Single-Prompt
**Claim**: Using 8 diverse prompts yields >2x semantic diversity vs. single prompt with 8x sampling.

**Test**: `multi_prompt_traces.json` vs. baseline with equivalent total samples.

### H3: Tree Search Finds New Correct Solutions
**Claim**: Tree-of-thought with width=3, depth=3 finds correct solutions that best-of-N misses.

**Test**: Measure "unique correct solutions" found by ToT vs. best-of-N with equal compute budget.

### H4: Diversity Transfers Across Domains
**Claim**: Techniques that improve diversity on MATH also improve diversity on HumanEval.

**Test**: Rank techniques by diversity gain on MATH, check if ranking correlates with HumanEval diversity.

### H5: Distilling Diverse Traces Improves Small Model Generalization
**Claim**: Training deepseek-coder-1.3b on maximally diverse CoT traces improves out-of-distribution math performance by >15%.

**Test**:
1. Create 3 datasets: (A) baseline traces, (B) diverse traces, (C) mixed
2. Fine-tune 3 copies of deepseek-coder-1.3b
3. Evaluate on held-out AIME 2025 and Minerva Math

---

## 7. Implementation Roadmap

### Week 1-2: Infrastructure
- [ ] Implement `probe/diverse_generation.py` (activation perturbation)
- [ ] Implement `probe/pca_diversity.py` (PCA-guided steering)
- [ ] Implement `eval/diversity_metrics.py` (all diversity metrics)
- [ ] Add code benchmarks support (`eval/code_benchmarks.py`)

### Week 3-4: Latent Space Experiments
- [ ] Run perturbation grid search (64 configs × 50 samples)
- [ ] Run PCA diversity experiments
- [ ] Analyze results, identify best layer/magnitude combinations

### Week 5-6: Width/Depth Search
- [ ] Implement `eval/clustered_sampling.py`
- [ ] Implement `eval/tree_of_thought.py`
- [ ] Implement `eval/multi_prompt_sample.py`
- [ ] Run all width/depth experiments

### Week 7: Integration
- [ ] Implement `eval/closed_source_generation.py` for GPT-4/Claude
- [ ] Generate diverse traces from closed-source models
- [ ] Combine best open-source + closed-source techniques

### Week 8: Cross-Domain Evaluation
- [ ] Run code benchmark evaluations
- [ ] Perform cross-benchmark analysis
- [ ] Generate final distillation dataset

### Week 9-10: Distillation & Validation
- [ ] Fine-tune deepseek-coder-1.3b on diverse traces
- [ ] Evaluate on held-out test sets
- [ ] Write up results

---

## 8. Expected Outcomes

### Deliverables
1. **Diversity-maximized dataset**: 50K+ diverse CoT traces for math+code
2. **Trained small model**: deepseek-coder-1.3b fine-tuned on diverse traces
3. **Method comparison**: Ranked list of diversity techniques by effectiveness
4. **Analysis paper**: "Maximizing Chain-of-Thought Diversity for Small Model Distillation"

### Success Metrics
- **Diversity gain**: >50% increase in semantic diversity vs. baseline
- **Correctness preservation**: >70% of baseline accuracy maintained
- **Generalization boost**: >15% accuracy gain on out-of-distribution tasks after distillation
- **Technique transferability**: Top-3 methods work across math AND code

---

## 9. Resource Requirements

### Compute
- **Training**: 4× A100 GPUs for fine-tuning (1 week)
- **Inference**: 8× A100 GPUs for diverse generation (2 weeks)
- **Estimated total**: ~5000 GPU-hours

### Storage
- **Traces**: ~100GB JSON (millions of samples)
- **Models**: ~50GB (checkpoints + LoRA adapters)

### API Costs (Closed-Source Models)
- GPT-4o: 50 questions × 50 samples × ~2000 tokens = ~$200
- Claude 3.5 Sonnet: Similar = ~$300
- **Total**: ~$500

---

## 10. Next Steps

**Immediate actions**:
1. Set up experiment tracking (use W&B or MLflow)
2. Implement diversity metrics first (needed for all experiments)
3. Start with latent space perturbation (fastest to implement)
4. Run small pilot (10 questions, 20 samples) to validate pipeline

**Start with this command**:
```bash
# Create experiment branch
git checkout -b diversity-experiments

# Install additional dependencies
pip install sentence-transformers scikit-learn wandb

# Run pilot experiment
python experiments/pilot_diversity.py --questions 10 --samples 20
```

Would you like me to implement any specific component first?
