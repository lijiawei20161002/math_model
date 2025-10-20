"""
Integration utilities for connecting the RL training framework
with the existing mathematical reasoning codebase.
"""

import logging
import re
import json
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class MathRewardFunction:
    """
    Reward function for mathematical reasoning tasks.

    Supports:
    - Outcome rewards: Check if the final answer is correct
    - Process rewards: Evaluate step-by-step reasoning (if reward model available)
    """

    def __init__(
        self,
        reward_model=None,
        use_outcome_reward: bool = True,
        use_process_reward: bool = False,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
    ):
        self.reward_model = reward_model
        self.use_outcome_reward = use_outcome_reward
        self.use_process_reward = use_process_reward
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward

    def extract_answer(self, text: str) -> Optional[str]:
        """
        Extract the final answer from a mathematical solution.

        Looks for patterns like:
        - "The answer is X"
        - "#### X"
        - "\\boxed{X}"
        """
        # Try boxed notation
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Try #### notation (common in MATH dataset)
        hash_match = re.search(r'####\s*(.+)', text)
        if hash_match:
            return hash_match.group(1).strip()

        # Try "the answer is" pattern
        answer_match = re.search(r'(?:the answer is|answer:)\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()

        # Try to find last number
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]

        return None

    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for comparison.
        """
        if answer is None:
            return ""

        # Remove common formatting
        answer = answer.strip()
        answer = answer.replace(",", "")
        answer = answer.replace("$", "")
        answer = answer.replace("\\", "")

        # Try to convert to number for numerical comparison
        try:
            num = float(answer)
            # Round to avoid floating point issues
            if abs(num - round(num)) < 1e-6:
                return str(int(round(num)))
            return str(round(num, 6))
        except ValueError:
            pass

        return answer.lower()

    def check_correctness(self, response: str, ground_truth: str) -> bool:
        """
        Check if response matches ground truth.
        """
        predicted = self.extract_answer(response)
        if predicted is None:
            return False

        predicted_norm = self.normalize_answer(predicted)
        truth_norm = self.normalize_answer(ground_truth)

        return predicted_norm == truth_norm

    def compute_outcome_reward(
        self,
        prompts: List[str],
        responses: List[str],
        ground_truths: List[str],
    ) -> List[float]:
        """
        Compute outcome-based rewards (correct/incorrect).

        Args:
            prompts: List of problem prompts
            responses: List of generated responses
            ground_truths: List of correct answers

        Returns:
            List of reward values
        """
        rewards = []
        for response, truth in zip(responses, ground_truths):
            is_correct = self.check_correctness(response, truth)
            reward = self.correct_reward if is_correct else self.incorrect_reward
            rewards.append(reward)

        return rewards

    def compute_process_reward(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[float]:
        """
        Compute process-based rewards using a reward model.

        Args:
            prompts: List of problem prompts
            responses: List of generated responses

        Returns:
            List of reward values
        """
        if self.reward_model is None:
            logger.warning("Process reward requested but no reward model provided")
            return [0.0] * len(responses)

        # TODO: Implement process reward computation
        # This would involve:
        # 1. Breaking down the solution into steps
        # 2. Scoring each step with the reward model
        # 3. Aggregating step-level rewards

        raise NotImplementedError("Process reward computation not yet implemented")

    def __call__(
        self,
        prompts: List[str],
        responses: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Compute rewards for responses.

        Args:
            prompts: List of problem prompts
            responses: List of generated responses
            ground_truths: List of correct answers (required for outcome rewards)

        Returns:
            List of reward values
        """
        if self.use_outcome_reward and ground_truths is None:
            raise ValueError("Ground truths required for outcome-based rewards")

        if self.use_outcome_reward:
            return self.compute_outcome_reward(prompts, responses, ground_truths)
        elif self.use_process_reward:
            return self.compute_process_reward(prompts, responses)
        else:
            # Default: zero rewards
            return [0.0] * len(responses)


class MathDataset(torch.utils.data.Dataset):
    """
    Dataset for mathematical reasoning problems.

    Supports loading from JSONL format with fields:
    - problem: The problem statement
    - solution: The solution (optional, for supervised learning)
    - answer: The correct answer
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[Any] = None,
        max_length: int = 2048,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.examples = self._load_data()

        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file"""
        examples = []

        # Try JSON format
        if self.data_path.endswith('.json'):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    examples = data
                else:
                    examples = [data]

        # Try JSONL format
        elif self.data_path.endswith('.jsonl'):
            with open(self.data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))

        else:
            logger.warning(f"Unknown file format: {self.data_path}")

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single example.

        Returns a dictionary with:
        - prompt: The problem prompt
        - answer: The correct answer
        - (optional) solution: The full solution
        """
        example = self.examples[idx]

        # Extract fields
        prompt = example.get('problem', example.get('question', ''))
        answer = example.get('answer', '')
        solution = example.get('solution', '')

        return {
            'prompt': prompt,
            'answer': answer,
            'solution': solution,
        }


def create_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
    torch_dtype = torch.bfloat16,
) -> AutoModelForCausalLM:
    """
    Load a model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint or HuggingFace model ID
        device: Device to load model on
        torch_dtype: Data type for model

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {checkpoint_path}")

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    return model


def create_tokenizer_from_checkpoint(
    checkpoint_path: str,
) -> AutoTokenizer:
    """
    Load a tokenizer from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint or HuggingFace model ID

    Returns:
        Loaded tokenizer
    """
    logger.info(f"Loading tokenizer from {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def format_problem_prompt(
    problem: str,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    instruction: str = "Solve the following mathematical problem. Show your work step by step.",
) -> str:
    """
    Format a mathematical problem as a prompt.

    Args:
        problem: The problem statement
        few_shot_examples: Optional list of example problem-solution pairs
        instruction: Instruction for the model

    Returns:
        Formatted prompt
    """
    prompt_parts = []

    # Add instruction
    if instruction:
        prompt_parts.append(instruction)
        prompt_parts.append("")

    # Add few-shot examples
    if few_shot_examples:
        for i, example in enumerate(few_shot_examples):
            prompt_parts.append(f"Problem {i+1}: {example['problem']}")
            prompt_parts.append(f"Solution: {example['solution']}")
            prompt_parts.append("")

    # Add the actual problem
    prompt_parts.append(f"Problem: {problem}")
    prompt_parts.append("Solution:")

    return "\n".join(prompt_parts)


def extract_metrics_from_responses(
    responses: List[str],
    ground_truths: List[str],
    reward_fn: MathRewardFunction,
) -> Dict[str, float]:
    """
    Extract evaluation metrics from model responses.

    Args:
        responses: List of generated responses
        ground_truths: List of correct answers
        reward_fn: Reward function for checking correctness

    Returns:
        Dictionary of metrics
    """
    num_correct = 0
    num_answered = 0
    num_total = len(responses)

    for response, truth in zip(responses, ground_truths):
        predicted = reward_fn.extract_answer(response)

        if predicted is not None:
            num_answered += 1

            if reward_fn.check_correctness(response, truth):
                num_correct += 1

    metrics = {
        "accuracy": num_correct / num_total if num_total > 0 else 0.0,
        "answer_rate": num_answered / num_total if num_total > 0 else 0.0,
        "num_correct": num_correct,
        "num_answered": num_answered,
        "num_total": num_total,
    }

    return metrics
