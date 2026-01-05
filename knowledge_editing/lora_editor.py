#!/usr/bin/env python3
"""
LoRA-based knowledge editing for mathematical heuristics.

Performs lightweight fine-tuning on synthetic heuristic examples
using LoRA (Low-Rank Adaptation) to install stable reasoning attractors.
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


@dataclass
class KnowledgeEditConfig:
    """Configuration for knowledge editing."""

    # Model
    model_name: str = "agentica-org/DeepScaleR-1.5B-Preview"
    output_dir: str = "./edited_model"

    # LoRA config
    lora_r: int = 8  # Rank
    lora_alpha: int = 16  # Scaling factor
    lora_dropout: float = 0.05
    target_modules: List[str] = None  # Will default to ["q_proj", "v_proj"]
    target_layers: Optional[List[int]] = None  # Restrict to specific layers (e.g., [10, 11, 12])

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_length: int = 512
    warmup_steps: int = 10

    # Data
    synthetic_data_path: str = "synthetic_heuristics.json"

    # Misc
    seed: int = 42
    use_8bit: bool = False  # Use 8-bit quantization

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class HeuristicDataset(Dataset):
    """Dataset for synthetic heuristic examples."""

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

        # Format: instruction + input -> output
        if "instruction" in item:
            prompt = f"{item['instruction']}\n\n{item['input']}\n\nSolution:"
            completion = item["output"]
        else:
            # Format: problem -> solution
            prompt = f"Problem: {item['problem']}\n\nSolution:"
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


class LoRAKnowledgeEditor:
    """Performs knowledge editing via LoRA fine-tuning."""

    def __init__(self, config: KnowledgeEditConfig):
        self.config = config
        self.model = None
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

        # Load model
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

        # Apply LoRA
        print("Applying LoRA...")
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(model, lora_config)
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
        """Perform knowledge editing via LoRA fine-tuning."""
        if self.model is None:
            self.load_model()

        # Load data
        data = self.load_synthetic_data()
        train_dataset = HeuristicDataset(
            data,
            self.tokenizer,
            max_length=self.config.max_length
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=10,
            save_strategy="epoch",
            fp16=torch.cuda.is_available() and not self.config.use_8bit,
            optim="adamw_torch",
            seed=self.config.seed,
            report_to="none",  # Disable WandB for now
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            ),
        )

        # Train
        print("\nStarting knowledge editing training...")
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


def edit_with_in_context_examples(
    model_name: str,
    synthetic_examples: List[Dict[str, str]],
    test_prompt: str,
    max_new_tokens: int = 512,
) -> str:
    """
    Pseudo-edit: inject synthetic examples in-context (no actual fine-tuning).

    This is the lightweight alternative to LoRA fine-tuning.

    Args:
        model_name: HuggingFace model name
        synthetic_examples: List of few-shot examples
        test_prompt: The problem to solve
        max_new_tokens: Max generation length

    Returns:
        Generated solution
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Build in-context prompt
    context = "Here are some examples of correct mathematical reasoning:\n\n"

    for i, ex in enumerate(synthetic_examples[:5], 1):  # Limit to 5 examples
        if "problem" in ex:
            context += f"Example {i}:\nProblem: {ex['problem']}\nSolution: {ex['solution']}\n\n"
        else:
            context += f"Example {i}:\n{ex['instruction']}\n{ex['input']}\nSolution: {ex['output']}\n\n"

    context += f"Now solve this problem:\nProblem: {test_prompt}\nSolution:"

    # Generate
    inputs = tokenizer(context, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
        )

    solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the generated part
    solution = solution[len(context):]

    return solution


def main():
    """CLI for knowledge editing."""
    import argparse

    parser = argparse.ArgumentParser(description="LoRA-based knowledge editing")
    parser.add_argument("--model", default="agentica-org/DeepScaleR-1.5B-Preview",
                       help="Model name or path")
    parser.add_argument("--data", default="synthetic_heuristics.json",
                       help="Path to synthetic training data")
    parser.add_argument("--output", default="./edited_model",
                       help="Output directory for edited model")
    parser.add_argument("--lora-r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--merge", action="store_true",
                       help="Merge LoRA weights and save full model")
    parser.add_argument("--merged-output", default="./edited_model_merged",
                       help="Output path for merged model")

    args = parser.parse_args()

    # Create config
    config = KnowledgeEditConfig(
        model_name=args.model,
        synthetic_data_path=args.data,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Create editor and train
    editor = LoRAKnowledgeEditor(config)
    editor.train()

    # Optionally merge
    if args.merge:
        editor.merge_and_save(args.merged_output)


if __name__ == "__main__":
    main()
