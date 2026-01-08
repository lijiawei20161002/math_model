#!/usr/bin/env python3
"""
Improved LoRA-based knowledge editing for mathematical heuristics (V2).

Key improvements over V1:
1. Targets both attention AND MLP layers (not just attention)
2. Layer-selective editing (middle-late layers only)
3. Higher LoRA rank for more capacity
4. Lower learning rate with longer warmup
5. Knowledge preservation via distillation loss
6. Validation on general math problems
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


@dataclass
class KnowledgeEditConfigV2:
    """Improved configuration for knowledge editing."""

    # Model
    model_name: str = "agentica-org/DeepScaleR-1.5B-Preview"
    output_dir: str = "./edited_model_v2"

    # LoRA config - IMPROVED DEFAULTS
    lora_r: int = 32  # Increased from 8 for more capacity
    lora_alpha: int = 64  # Scaled with rank
    lora_dropout: float = 0.05

    # Target BOTH attention and MLP layers
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Full attention
        "gate_proj", "up_proj", "down_proj"      # MLP layers
    ])

    # Layer-selective: target middle-to-late layers
    # For 24-layer model: layers 12-19
    # For 32-layer model: layers 16-26
    target_layers: Optional[List[int]] = field(default_factory=lambda: list(range(12, 20)))

    # Training - GENTLER DEFAULTS
    num_train_epochs: int = 5  # More epochs at lower LR
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5  # Much lower than 2e-4
    max_length: int = 512
    warmup_steps: int = 50  # Longer warmup
    weight_decay: float = 0.01  # L2 regularization

    # Knowledge preservation
    use_preservation_loss: bool = True  # Enable KD loss
    preservation_alpha: float = 0.3  # Weight for preservation loss

    # Validation on general math
    eval_on_general_math: bool = True
    eval_steps: int = 50
    general_math_dataset: str = "gsm8k"  # Or "math" subset

    # Data
    synthetic_data_path: str = "synthetic_heuristics_v2.json"

    # Misc
    seed: int = 42
    use_8bit: bool = False
    gradient_checkpointing: bool = True  # Enable for larger effective batch


class HeuristicDatasetV2(Dataset):
    """Improved dataset with better formatting."""

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Improved prompt formatting with explicit structure
        if "instruction" in item:
            prompt = f"<|user|>\n{item['instruction']}\n\n{item['input']}\n\n<|assistant|>\nSolution:"
            completion = item["output"]
        else:
            prompt = f"<|user|>\nProblem: {item['problem']}\n\n<|assistant|>\nSolution:"
            completion = item["solution"]

        # Tokenize
        full_text = prompt + "\n" + completion
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Create labels (only predict completion part)
        prompt_ids = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]

        labels = tokenized["input_ids"].clone()
        # Mask prompt tokens with -100 (ignore in loss)
        labels[0, : len(prompt_ids)] = -100

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


class PreservationTrainer(Trainer):
    """Custom trainer with knowledge preservation loss."""

    def __init__(self, *args, base_model=None, preservation_alpha=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = base_model
        self.preservation_alpha = preservation_alpha

        # Freeze base model
        if self.base_model is not None:
            self.base_model.eval()
            for param in self.base_model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with optional preservation component."""
        labels = inputs.pop("labels")

        # Forward pass on student (edited) model
        outputs = model(**inputs)
        student_logits = outputs.logits

        # Standard cross-entropy loss
        heuristic_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # Knowledge preservation loss (KL divergence from base model)
        if self.base_model is not None and self.preservation_alpha > 0:
            with torch.no_grad():
                teacher_outputs = self.base_model(**inputs)
                teacher_logits = teacher_outputs.logits

            # Only compute KL on non-masked tokens
            mask = (labels != -100).unsqueeze(-1).expand_as(student_logits)

            student_probs = F.log_softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)

            # KL divergence
            kl_div = F.kl_div(
                student_probs[mask].view(-1, student_logits.size(-1)),
                teacher_probs[mask].view(-1, teacher_logits.size(-1)),
                reduction='batchmean'
            )

            # Combined loss
            total_loss = (1 - self.preservation_alpha) * heuristic_loss + \
                        self.preservation_alpha * kl_div
        else:
            total_loss = heuristic_loss

        return (total_loss, outputs) if return_outputs else total_loss


class GeneralMathValidationCallback(TrainerCallback):
    """Callback to validate on general math problems during training."""

    def __init__(self, eval_dataset, tokenizer, log_interval=50):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.log_interval = log_interval

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Evaluate every N steps."""
        if state.global_step % self.log_interval == 0:
            print(f"\n[Step {state.global_step}] Validating on general math...")

            # TODO: Implement actual evaluation
            # For now, just log a placeholder
            print("  → General math validation not yet implemented")


class LoRAKnowledgeEditorV2:
    """Improved knowledge editor with preservation mechanisms."""

    def __init__(self, config: KnowledgeEditConfigV2):
        self.config = config
        self.model = None
        self.base_model = None  # Keep base model for preservation
        self.tokenizer = None
        self.trainer = None

    def load_model(self):
        """Load the base model and apply LoRA."""
        print(f"Loading model: {self.config.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model (for preservation loss)
        if self.config.use_preservation_loss:
            print("Loading base model for knowledge preservation...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.base_model.eval()

        # Load model to edit
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }
        if self.config.use_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )

        if self.config.use_8bit:
            model = prepare_model_for_kbit_training(model)

        # Apply LoRA with improved config
        print("Applying LoRA...")
        print(f"  Rank: {self.config.lora_r}")
        print(f"  Alpha: {self.config.lora_alpha}")
        print(f"  Target modules: {self.config.target_modules}")
        print(f"  Target layers: {self.config.target_layers}")

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            layers_to_transform=self.config.target_layers,  # Layer-selective
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(model, lora_config)
        print("\nTrainable parameters:")
        self.model.print_trainable_parameters()

        return self.model, self.tokenizer

    def load_synthetic_data(self) -> List[Dict[str, str]]:
        """Load synthetic heuristic training data."""
        print(f"Loading data from: {self.config.synthetic_data_path}")
        with open(self.config.synthetic_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} examples")
        return data

    def train(self):
        """Perform knowledge editing via LoRA fine-tuning with preservation."""
        if self.model is None:
            self.load_model()

        # Load data
        data = self.load_synthetic_data()
        train_dataset = HeuristicDatasetV2(
            data,
            self.tokenizer,
            max_length=self.config.max_length
        )

        # Training arguments with improved defaults
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="steps" if self.config.eval_on_general_math else "no",
            eval_steps=self.config.eval_steps if self.config.eval_on_general_math else None,
            load_best_model_at_end=self.config.eval_on_general_math,
            gradient_checkpointing=self.config.gradient_checkpointing,
            fp16=torch.cuda.is_available() and not self.config.use_8bit,
            optim="adamw_torch",
            seed=self.config.seed,
            report_to="none",
        )

        # Create trainer with preservation
        trainer_class = PreservationTrainer if self.config.use_preservation_loss else Trainer

        trainer_kwargs = {
            "model": self.model,
            "args": training_args,
            "train_dataset": train_dataset,
            "data_collator": DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            ),
        }

        if self.config.use_preservation_loss:
            trainer_kwargs["base_model"] = self.base_model
            trainer_kwargs["preservation_alpha"] = self.config.preservation_alpha

        self.trainer = trainer_class(**trainer_kwargs)

        # Train
        print("\n" + "="*70)
        print("Starting knowledge editing training...")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Epochs: {self.config.num_train_epochs}")
        print(f"  Warmup steps: {self.config.warmup_steps}")
        print(f"  Preservation loss: {self.config.use_preservation_loss}")
        if self.config.use_preservation_loss:
            print(f"  Preservation alpha: {self.config.preservation_alpha}")
        print("="*70 + "\n")

        self.trainer.train()

        # Save
        self.save()

        return self.model

    def save(self):
        """Save the edited model."""
        print(f"\nSaving edited model to: {self.config.output_dir}")
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print("Model saved successfully!")

    def merge_and_save(self, output_path: str):
        """Merge LoRA weights back into base model and save."""
        print(f"Merging LoRA weights and saving to: {output_path}")

        # Merge LoRA
        merged_model = self.model.merge_and_unload()

        # Save merged model
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("Merged model saved successfully!")


def main():
    """CLI for improved knowledge editing."""
    import argparse

    parser = argparse.ArgumentParser(description="Improved LoRA-based knowledge editing (V2)")
    parser.add_argument("--model", default="agentica-org/DeepScaleR-1.5B-Preview",
                       help="Model name or path")
    parser.add_argument("--data", default="synthetic_heuristics_v2.json",
                       help="Path to synthetic training data")
    parser.add_argument("--output", default="./edited_model_v2",
                       help="Output directory for edited model")

    # LoRA config
    parser.add_argument("--lora-r", type=int, default=32,
                       help="LoRA rank (default: 32, improved from 8)")
    parser.add_argument("--lora-alpha", type=int, default=64,
                       help="LoRA alpha (default: 64)")
    parser.add_argument("--target-layers", type=str, default="12-20",
                       help="Layer range to target (e.g., '12-20')")

    # Training config
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate (default: 5e-5, lowered from 2e-4)")
    parser.add_argument("--warmup-steps", type=int, default=50,
                       help="Warmup steps")

    # Preservation
    parser.add_argument("--no-preservation", action="store_true",
                       help="Disable knowledge preservation loss")
    parser.add_argument("--preservation-alpha", type=float, default=0.3,
                       help="Weight for preservation loss (0-1)")

    # Output
    parser.add_argument("--merge", action="store_true",
                       help="Merge LoRA weights and save full model")
    parser.add_argument("--merged-output", default="./edited_model_v2_merged",
                       help="Output path for merged model")

    args = parser.parse_args()

    # Parse layer range
    if "-" in args.target_layers:
        start, end = map(int, args.target_layers.split("-"))
        target_layers = list(range(start, end))
    else:
        target_layers = [int(args.target_layers)]

    # Create config
    config = KnowledgeEditConfigV2(
        model_name=args.model,
        synthetic_data_path=args.data,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_layers=target_layers,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        use_preservation_loss=not args.no_preservation,
        preservation_alpha=args.preservation_alpha,
    )

    print("\n" + "="*70)
    print("IMPROVED KNOWLEDGE EDITING (V2)")
    print("="*70)
    print(f"Model: {config.model_name}")
    print(f"Data: {config.synthetic_data_path}")
    print(f"Output: {config.output_dir}")
    print(f"\nLoRA Config:")
    print(f"  Rank: {config.lora_r}")
    print(f"  Alpha: {config.lora_alpha}")
    print(f"  Target layers: {config.target_layers}")
    print(f"\nTraining Config:")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Warmup: {config.warmup_steps}")
    print(f"\nPreservation:")
    print(f"  Enabled: {config.use_preservation_loss}")
    if config.use_preservation_loss:
        print(f"  Alpha: {config.preservation_alpha}")
    print("="*70 + "\n")

    # Create editor and train
    editor = LoRAKnowledgeEditorV2(config)
    editor.train()

    # Optionally merge
    if args.merge:
        editor.merge_and_save(args.merged_output)


if __name__ == "__main__":
    main()
