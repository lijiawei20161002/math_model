# RL Training Framework for Mathematical Reasoning

A comprehensive reinforcement learning training framework for mathematical reasoning models, featuring implementations of **M2PO**, **GRPO**, and **PPO** algorithms.

## Overview

This framework provides production-ready implementations of state-of-the-art RL algorithms specifically optimized for training language models on mathematical reasoning tasks.

### Supported Algorithms

- **M2PO (Mathematical Measure Optimization for Policy Optimization)**: Uses second-order KL constraints (M2/KL² budget) for adaptive per-token clipping
- **GRPO (Group Relative Policy Optimization)**: Group-based advantage estimation using RLOO (Reinforce Leave One Out)
- **PPO (Proximal Policy Optimization)**: Standard implementation with GAE and clipped surrogate objective

## Features

- ✅ **Three SOTA RL algorithms** (M2PO, GRPO, PPO)
- ✅ **Distributed training** support (DDP, DeepSpeed)
- ✅ **Flexible reward functions** (outcome-based, process-based)
- ✅ **Reference model** support for KL penalty
- ✅ **Value function** training
- ✅ **Comprehensive logging** (WandB, TensorBoard)
- ✅ **Checkpoint management** and resumption
- ✅ **Evaluation** during training

## Installation

```bash
cd training
pip install -r requirements.txt
```

### Core Dependencies

- `torch >= 2.0.0`
- `transformers >= 4.30.0`
- `accelerate >= 0.20.0`
- `wandb >= 0.15.0` (optional, for logging)

## Quick Start

### 1. Basic M2PO Training

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trainer.rl_trainer import RLTrainer, RLTrainerConfig

# Load model
model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Configure M2PO
config = RLTrainerConfig(
    algorithm="m2po",
    m2po_config={
        "m2_budget": 0.01,
        "miniclip_low": 0.3,
        "miniclip_high": 0.5,
    },
    num_train_epochs=3,
    batch_size=4,
    learning_rate=1e-5,
    output_dir="./outputs/m2po",
)

# Create trainer
trainer = RLTrainer(
    model=model,
    config=config,
    tokenizer=tokenizer,
    reward_fn=your_reward_function,
    train_dataset=train_dataset,
)

# Train
trainer.train()
```

### 2. Using Configuration Files

```bash
# Train with M2PO
python scripts/train.py --config configs/m2po_config.yaml --model_path <model>

# Train with GRPO
python scripts/train.py --config configs/grpo_config.yaml --model_path <model>

# Train with PPO
python scripts/train.py --config configs/ppo_config.yaml --model_path <model>
```

### 3. Distributed Training

```bash
# Single node, 8 GPUs
torchrun --nproc_per_node=8 scripts/train_distributed.py \
    --config configs/m2po_config.yaml \
    --model_path <model>

# Multiple nodes
torchrun --nnodes=4 --nproc_per_node=8 \
    --node_rank=0 --master_addr=<addr> --master_port=<port> \
    scripts/train_distributed.py --config configs/m2po_config.yaml
```

## Algorithm Comparison

| Feature | PPO | M2PO | GRPO |
|---------|-----|------|------|
| **Clipping** | Fixed ratio | Adaptive (M2 budget) | Fixed ratio |
| **Advantages** | GAE + Value Fn | GAE + Value Fn | Group-relative (RLOO) |
| **Value Function** | Required | Required | Optional |
| **Best For** | General RL | Math reasoning | Batch generation |
| **Computational Cost** | Medium | Medium-High | Low-Medium |

### When to Use Each Algorithm

**Use M2PO if:**
- You're working on mathematical reasoning
- You want adaptive, data-driven clipping
- You need precise control over policy updates
- You can tune the M2 budget hyperparameter

**Use GRPO if:**
- You can generate multiple responses per prompt
- You want to avoid training a value function
- You have efficient batch evaluation
- Variance reduction via group statistics is desirable

**Use PPO if:**
- You want a stable, well-tested baseline
- You're doing initial experimentation
- You need predictable behavior

## M2PO Details

M2PO (Mathematical Measure Optimization for Policy Optimization) uses a second-order KL constraint to adaptively control policy updates.

### Key Concepts

1. **M2 Budget**: Controls the allowed KL² divergence per harmful token
   - Smaller budget → more conservative updates
   - Typical range: 0.005 - 0.02

2. **Harmful Tokens**: Tokens where the policy update would hurt the objective
   - Positive advantage with ratio > 1
   - Negative advantage with ratio < 1

3. **Adaptive Clipping**: Dynamically computes clip bounds based on M2 budget
   - Automatically adjusts to data distribution
   - More flexible than PPO's fixed clipping

### M2PO Hyperparameters

```python
m2po_config = {
    "m2_budget": 0.01,         # M2 (KL²) budget per harmful token
    "miniclip_low": 0.3,       # Minimum clipping for ratio < 1
    "miniclip_high": 0.5,      # Minimum clipping for ratio > 1
    "loss_agg_mode": "token-mean",  # Loss aggregation mode
}
```

**Tuning Tips:**
- Start with `m2_budget=0.01` and adjust based on training stability
- Lower M2 budget if training is unstable (0.005-0.01)
- Higher M2 budget for faster learning (0.015-0.02), but watch for instability
- `miniclip_low/high` provide minimum clipping when M2 budget isn't violated

### M2PO Algorithm Flow

```
1. Generate rollouts from current policy
2. Compute advantages (e.g., using GAE)
3. Identify "harmful tokens" (updates that hurt objective)
4. Sort harmful tokens by KL² divergence
5. Find threshold τ such that capping KL at τ yields total KL² ≤ budget
6. Map τ to adaptive clip bounds: [exp(-τ), exp(+τ)]
7. Compute clipped policy gradient loss
8. Update policy
```

## Directory Structure

```
training/
├── algorithms/           # RL algorithm implementations
│   ├── m2po.py          # M2PO algorithm
│   ├── grpo.py          # GRPO algorithm
│   ├── ppo.py           # PPO algorithm
│   └── utils.py         # Shared utilities
├── trainer/             # Training framework
│   ├── base_trainer.py  # Base trainer class
│   └── rl_trainer.py    # RL-specific trainer
├── configs/             # Configuration files
│   ├── m2po_config.yaml
│   ├── grpo_config.yaml
│   └── ppo_config.yaml
├── scripts/             # Training scripts
│   ├── train.py         # Single-node training
│   ├── train_distributed.py  # Multi-node training
│   └── evaluate.py      # Evaluation script
├── examples/            # Example scripts
│   ├── train_m2po_example.py
│   └── compare_algorithms.py
├── utils/              # Utilities
│   └── monitoring.py   # Logging and monitoring
└── requirements.txt    # Dependencies
```

## Configuration

### Core Configuration Options

```yaml
# Model
model_name: "math_model_m2po"
model_path: null  # Path to checkpoint

# Algorithm
algorithm: "m2po"  # Options: "ppo", "m2po", "grpo"

# M2PO Config
m2po_config:
  m2_budget: 0.01
  miniclip_low: 0.3
  miniclip_high: 0.5
  loss_agg_mode: "token-mean"

# Training
num_train_epochs: 3
batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
warmup_steps: 100

# RL-specific
rollout_batch_size: 32
ppo_epochs: 4
mini_batch_size: 8

# Generation
max_new_tokens: 512
temperature: 1.0
top_k: 50
top_p: 0.95
do_sample: true

# Reward
use_outcome_reward: true
use_process_reward: false

# Reference model (for KL penalty)
use_reference_model: true
reference_model_path: null

# Logging
logging_steps: 10
eval_steps: 100
save_steps: 500
use_wandb: false
```

## Reward Functions

The framework supports flexible reward functions. Here's an example for mathematical reasoning:

```python
def math_reward_fn(prompts, responses, ground_truths):
    """
    Compute rewards based on mathematical correctness.

    Args:
        prompts: List of problem prompts
        responses: List of generated solutions
        ground_truths: List of correct answers

    Returns:
        Tensor of rewards
    """
    rewards = []
    for response, truth in zip(responses, ground_truths):
        # Extract answer from response
        predicted_answer = extract_answer(response)

        # Check correctness
        if is_mathematically_equivalent(predicted_answer, truth):
            reward = 1.0
        else:
            reward = 0.0

        # Optional: Add intermediate rewards
        if has_valid_reasoning_steps(response):
            reward += 0.2

        rewards.append(reward)

    return torch.tensor(rewards)
```

## Examples

### Compare Algorithms

Run the comparison script to see detailed differences:

```bash
python examples/compare_algorithms.py
```

### Train with M2PO

```bash
python examples/train_m2po_example.py
```

### Evaluate Model

```bash
python scripts/evaluate.py \
    --model_path ./outputs/m2po/final_model \
    --data_path ./data/test.json \
    --output_path ./results.json
```

## Monitoring and Logging

### Weights & Biases (WandB)

```yaml
use_wandb: true
wandb_project: "math-reasoning"
wandb_entity: "your-entity"
wandb_run_name: "m2po-experiment-1"
```

### TensorBoard

```bash
tensorboard --logdir ./outputs/m2po/tensorboard
```

### Key Metrics

The framework logs:

**Policy Metrics:**
- `policy/loss`: Policy gradient loss
- `policy/clip_fraction`: Fraction of clipped tokens
- `policy/entropy`: Policy entropy
- `policy/kl`: KL divergence from reference

**M2PO-Specific:**
- `m2po/M2`: Current M2 (KL²) value
- `m2po/M2_after`: Expected M2 after clipping
- `m2po/M2_budget`: Configured budget
- `m2po/clip_low`: Lower clip bound
- `m2po/clip_high`: Upper clip bound

**Ratio Statistics:**
- `ratio_pos/*`: Ratio distribution for positive advantages
- `ratio_neg/*`: Ratio distribution for negative advantages
- `ratio_nonzero/avg`: Average ratio for non-zero advantages

**Rewards:**
- `rewards/mean`: Average reward
- `rewards/std`: Reward standard deviation
- `rewards/max`: Maximum reward
- `rewards/min`: Minimum reward

**Training:**
- `train/learning_rate`: Current learning rate
- `train/grad_norm`: Gradient norm
- `train/loss`: Total loss

## Advanced Usage

### Custom Algorithm Configuration

```python
# Create custom M2PO config
from algorithms.m2po import M2POConfig

custom_config = M2POConfig(
    m2_budget=0.015,
    miniclip_low=0.2,
    miniclip_high=0.4,
    loss_agg_mode="seq-mean-token-sum",
)

# Use in trainer
config = RLTrainerConfig(
    algorithm="m2po",
    m2po_config=custom_config.__dict__,
    # ... other options
)
```

### Multi-Stage Training

```python
# Stage 1: Warm-up with PPO
ppo_config = RLTrainerConfig(algorithm="ppo", ...)
ppo_trainer = RLTrainer(model, ppo_config, ...)
ppo_trainer.train()

# Stage 2: Fine-tune with M2PO
m2po_config = RLTrainerConfig(algorithm="m2po", ...)
m2po_trainer = RLTrainer(model, m2po_config, ...)
m2po_trainer.train()
```

### Using Process Rewards

```python
config = RLTrainerConfig(
    use_outcome_reward=True,
    use_process_reward=True,  # Enable process rewards
    reward_model_path="path/to/process_reward_model",
    # ...
)
```

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
- Reduce `batch_size` or `rollout_batch_size`
- Increase `gradient_accumulation_steps`
- Use smaller `max_new_tokens`
- Enable gradient checkpointing

**2. Training Instability**
- Lower the learning rate
- Reduce M2 budget (for M2PO): try 0.005
- Increase warmup steps
- Check reward function for extreme values

**3. Slow Convergence**
- Increase learning rate cautiously
- Adjust M2 budget (for M2PO): try 0.015-0.02
- Check if rewards are sparse (add intermediate rewards)
- Verify advantage computation

**4. KL Divergence Exploding**
- Enable reference model: `use_reference_model=True`
- Add KL penalty: set `kl_coef > 0`
- For M2PO: lower M2 budget
- For PPO: lower clip ratio

## Performance Tips

1. **Batch Size**: Use largest possible batch that fits in memory
2. **Gradient Accumulation**: Multiply effective batch size without OOM
3. **Mixed Precision**: Use `torch.cuda.amp` for faster training
4. **Distributed**: Scale to multiple GPUs/nodes for large models
5. **Checkpointing**: Save frequently to recover from failures

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- M2PO implementation based on the original M2PO paper and codebase
- GRPO implementation inspired by RLOO and group-based RL methods
- PPO implementation follows the original Schulman et al. paper

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to examples/ for usage patterns
