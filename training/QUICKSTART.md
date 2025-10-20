# Quick Start Guide: M2PO Training

Get started with M2PO training in 5 minutes!

## Prerequisites

```bash
# Install dependencies
cd training
pip install -r requirements.txt
```

## Option 1: Run the Example (Fastest)

```bash
# Run the M2PO example with a small model
python examples/train_m2po_example.py
```

This will:
- Load a small GPT-2 model for testing
- Create a synthetic math dataset
- Train for 3 epochs with M2PO
- Save the model to `./outputs/m2po_example/`

## Option 2: Use Your Own Model

### Step 1: Prepare Your Data

Create a JSON file with your math problems:

```json
[
  {
    "prompt": "What is 15 + 27?",
    "answer": "42"
  },
  {
    "prompt": "Solve: 2x + 5 = 13",
    "answer": "x = 4"
  }
]
```

### Step 2: Create Training Script

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trainer.rl_trainer import RLTrainer, RLTrainerConfig
import torch
import json

# Load your model
model = AutoModelForCausalLM.from_pretrained("your-model-name")
tokenizer = AutoTokenizer.from_pretrained("your-model-name")

# Load data
with open("your_data.json") as f:
    train_data = json.load(f)

# Define reward function
def reward_fn(prompts, responses, ground_truths=None):
    # Your reward logic here
    # Return torch.Tensor of rewards
    rewards = []
    for response in responses:
        # Check if answer is correct
        reward = 1.0 if is_correct(response) else 0.0
        rewards.append(reward)
    return torch.tensor(rewards)

# Configure M2PO
config = RLTrainerConfig(
    algorithm="m2po",
    m2po_config={
        "m2_budget": 0.01,      # Start with this, tune if needed
        "miniclip_low": 0.3,
        "miniclip_high": 0.5,
    },

    # Training params
    num_train_epochs=3,
    batch_size=4,
    learning_rate=1e-5,

    # Output
    output_dir="./outputs/my_m2po_model",
)

# Create and run trainer
trainer = RLTrainer(
    model=model,
    config=config,
    tokenizer=tokenizer,
    reward_fn=reward_fn,
    train_dataset=train_data,
)

trainer.train()
```

### Step 3: Run It

```bash
python your_training_script.py
```

## Option 3: Use Config Files

### Step 1: Edit Config

Edit `configs/m2po_config.yaml`:

```yaml
# Your model
model_path: "path/to/your/model"

# M2PO settings
algorithm: "m2po"
m2po_config:
  m2_budget: 0.01
  miniclip_low: 0.3
  miniclip_high: 0.5

# Training
num_train_epochs: 3
batch_size: 4
learning_rate: 1.0e-5

# Output
output_dir: "./outputs/my_experiment"
```

### Step 2: Run Training

```bash
python scripts/train.py --config configs/m2po_config.yaml
```

## Understanding M2PO Parameters

### `m2_budget` (Most Important!)

Controls how much the policy can change. Think of it as a "safety budget."

- **Lower (0.005-0.01)**: More conservative, more stable
- **Higher (0.015-0.02)**: More aggressive, faster learning, less stable

**Start with 0.01** and adjust:
- If training is unstable → decrease to 0.005
- If learning is too slow → increase to 0.015

### `miniclip_low` and `miniclip_high`

Minimum clipping values (like PPO's clip ratio).

- Usually keep at 0.3 and 0.5
- Only change if you understand M2PO deeply

### `loss_agg_mode`

How to aggregate loss across tokens:
- `"token-mean"`: Average across all tokens (default, recommended)
- `"seq-mean-token-sum"`: Sum per sequence, then average sequences

## Monitoring Your Training

### Check Progress

The trainer logs:
- `m2po/M2`: Current M2 value (should be ≤ budget)
- `m2po/clip_low` and `m2po/clip_high`: Adaptive clip bounds
- `rewards/mean`: Average reward (should increase)
- `policy/kl`: KL divergence (should be controlled)

### Is It Working?

Good signs:
- ✅ M2 value stays near budget
- ✅ Rewards increase over time
- ✅ KL divergence controlled
- ✅ Clip bounds adapt to data

Bad signs:
- ❌ M2 value much lower than budget → increase budget
- ❌ Rewards not increasing → check reward function
- ❌ KL divergence exploding → decrease budget or learning rate
- ❌ Loss is NaN → decrease learning rate

## Troubleshooting

### "Out of Memory"
```python
config = RLTrainerConfig(
    batch_size=2,  # Reduce this
    gradient_accumulation_steps=4,  # Increase this
    # ...
)
```

### "Training is Unstable"
```python
config = RLTrainerConfig(
    m2po_config={
        "m2_budget": 0.005,  # Decrease budget
    },
    learning_rate=5e-6,  # Lower learning rate
    warmup_steps=200,  # More warmup
    # ...
)
```

### "Training is Too Slow"
```python
config = RLTrainerConfig(
    m2po_config={
        "m2_budget": 0.015,  # Increase budget
    },
    learning_rate=2e-5,  # Higher learning rate
    # ...
)
```

## Next Steps

1. **Read the full README**: `training/README.md`
2. **Compare algorithms**: `python examples/compare_algorithms.py`
3. **Tune hyperparameters**: Start with M2 budget
4. **Monitor training**: Use WandB or TensorBoard
5. **Evaluate**: `python scripts/evaluate.py --model_path <path>`

## Common Use Cases

### Fine-tune a Math Model

```python
config = RLTrainerConfig(
    algorithm="m2po",
    m2po_config={"m2_budget": 0.01},
    num_train_epochs=3,
    learning_rate=1e-5,
    use_outcome_reward=True,
)
```

### Train from Scratch (Larger Budget)

```python
config = RLTrainerConfig(
    algorithm="m2po",
    m2po_config={"m2_budget": 0.02},  # Higher for from-scratch
    num_train_epochs=10,
    learning_rate=3e-5,
)
```

### Conservative Fine-tuning (Preserve Base Model)

```python
config = RLTrainerConfig(
    algorithm="m2po",
    m2po_config={"m2_budget": 0.005},  # Very conservative
    use_reference_model=True,  # KL penalty vs base model
    kl_coef=0.1,
    learning_rate=5e-6,
)
```

## Getting Help

- **Check examples**: `examples/train_m2po_example.py`
- **Read docs**: `README.md`
- **Compare algorithms**: `examples/compare_algorithms.py`
- **Check logs**: Look for M2PO-specific metrics

Happy training! 🚀
