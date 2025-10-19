"""
API clients for Claude (Anthropic) and OpenAI models to generate diverse reasoning paths.
"""

import os
import asyncio
from typing import List, Dict, Optional
from anthropic import Anthropic
from openai import OpenAI


class DiverseReasoningGenerator:
    """Generate diverse reasoning paths using multiple LLM providers."""

    def __init__(self, anthropic_api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        """
        Initialize API clients.

        Args:
            anthropic_api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if self.anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
        else:
            self.anthropic_client = None
            print("Warning: Anthropic API key not found. Claude models will be unavailable.")

        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            print("Warning: OpenAI API key not found. GPT models will be unavailable.")

    def _create_prompt_variants(self, problem: str) -> List[Dict[str, str]]:
        """
        Create diverse prompt templates to encourage different reasoning approaches.

        Returns:
            List of prompt variants with metadata
        """
        return [
            {
                "name": "Algebraic",
                "prompt": f"""Solve this problem using algebraic methods (equations, substitution, etc.):

{problem}

Provide a step-by-step solution and clearly state your final answer.""",
                "category": "systematic"
            },
            {
                "name": "Trial and Error",
                "prompt": f"""Solve this problem by trying different possibilities systematically:

{problem}

Show each attempt you make and explain your reasoning. State your final answer clearly.""",
                "category": "empirical"
            },
            {
                "name": "Pattern Recognition",
                "prompt": f"""Solve this problem by identifying patterns and relationships:

{problem}

Look for patterns in the numbers and constraints. Show your reasoning and final answer.""",
                "category": "heuristic"
            },
            {
                "name": "Backward Chaining",
                "prompt": f"""Solve this problem by working backwards from the goal:

{problem}

Start with what you need to find and work backwards. Show your steps and final answer.""",
                "category": "systematic"
            },
            {
                "name": "Visual/Diagrammatic",
                "prompt": f"""Solve this problem using visual or diagrammatic reasoning:

{problem}

Describe any diagrams or visual representations you would use. Show your reasoning and final answer.""",
                "category": "visual"
            },
            {
                "name": "Estimation First",
                "prompt": f"""Solve this problem by first estimating, then calculating precisely:

{problem}

Start with a rough estimate, then refine to get the exact answer. Show your work.""",
                "category": "approximate"
            },
        ]

    def generate_with_claude(
        self,
        problem: str,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 1.0,
        max_tokens: int = 2000,
    ) -> Dict[str, any]:
        """
        Generate a reasoning path using Claude.

        Args:
            problem: The math problem to solve
            model: Claude model to use
            temperature: Sampling temperature (higher = more diverse)
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with reasoning steps and answer
        """
        if not self.anthropic_client:
            raise ValueError("Anthropic API key not configured")

        try:
            message = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": problem
                }]
            )

            response_text = message.content[0].text
            return {
                "response": response_text,
                "model": model,
                "temperature": temperature,
            }
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return {
                "response": "",
                "model": model,
                "error": str(e)
            }

    def generate_with_openai(
        self,
        problem: str,
        model: str = "gpt-4o",
        temperature: float = 1.0,
        max_tokens: int = 2000,
    ) -> Dict[str, any]:
        """
        Generate a reasoning path using OpenAI GPT.

        Args:
            problem: The math problem to solve
            model: OpenAI model to use
            temperature: Sampling temperature (higher = more diverse)
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with reasoning steps and answer
        """
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": problem
                }],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            response_text = response.choices[0].message.content
            return {
                "response": response_text,
                "model": model,
                "temperature": temperature,
            }
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return {
                "response": "",
                "model": model,
                "error": str(e)
            }

    def generate_diverse_paths(
        self,
        problem: str,
        num_claude_samples: int = 3,
        num_openai_samples: int = 3,
        claude_model: str = "claude-3-5-sonnet-20241022",
        openai_model: str = "gpt-4o",
        temperature_range: tuple = (0.7, 1.3),
    ) -> List[Dict[str, any]]:
        """
        Generate diverse reasoning paths using multiple models and prompt strategies.

        Args:
            problem: The math problem to solve
            num_claude_samples: Number of samples to generate with Claude
            num_openai_samples: Number of samples to generate with OpenAI
            claude_model: Claude model to use
            openai_model: OpenAI model to use
            temperature_range: (min, max) temperature for diversity

        Returns:
            List of reasoning path dictionaries
        """
        results = []
        prompt_variants = self._create_prompt_variants(problem)

        # Generate samples with Claude
        if self.anthropic_client and num_claude_samples > 0:
            for i in range(num_claude_samples):
                # Use different prompt variants
                variant = prompt_variants[i % len(prompt_variants)]
                # Vary temperature for additional diversity
                temp = temperature_range[0] + (temperature_range[1] - temperature_range[0]) * (i / max(num_claude_samples - 1, 1))

                result = self.generate_with_claude(
                    problem=variant["prompt"],
                    model=claude_model,
                    temperature=temp,
                )
                result["prompt_variant"] = variant["name"]
                result["category"] = variant["category"]
                result["provider"] = "anthropic"
                results.append(result)

        # Generate samples with OpenAI
        if self.openai_client and num_openai_samples > 0:
            for i in range(num_openai_samples):
                # Use different prompt variants
                variant = prompt_variants[i % len(prompt_variants)]
                # Vary temperature for additional diversity
                temp = temperature_range[0] + (temperature_range[1] - temperature_range[0]) * (i / max(num_openai_samples - 1, 1))

                result = self.generate_with_openai(
                    problem=variant["prompt"],
                    model=openai_model,
                    temperature=temp,
                )
                result["prompt_variant"] = variant["name"]
                result["category"] = variant["category"]
                result["provider"] = "openai"
                results.append(result)

        return results

    def parse_reasoning_steps(self, response: str) -> List[str]:
        """
        Parse a model response into individual reasoning steps.

        Args:
            response: The full model response

        Returns:
            List of reasoning steps
        """
        # Split by common step markers
        lines = response.split('\n')
        steps = []
        current_step = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a new step (numbered, bulleted, or "Step X:")
            is_new_step = (
                line[0].isdigit() and ('. ' in line[:5] or ') ' in line[:5]) or
                line.startswith('- ') or
                line.startswith('* ') or
                line.lower().startswith('step ')
            )

            if is_new_step and current_step:
                steps.append(' '.join(current_step))
                current_step = [line]
            else:
                current_step.append(line)

        if current_step:
            steps.append(' '.join(current_step))

        # If no clear steps found, split into sentences
        if len(steps) < 2:
            import re
            sentences = re.split(r'[.!?]+', response)
            steps = [s.strip() for s in sentences if s.strip()]

        return steps[:10]  # Limit to 10 steps for visualization

    def extract_final_answer(self, response: str) -> Optional[int]:
        """
        Extract the final numerical answer from a response.

        Args:
            response: The model response

        Returns:
            The extracted answer as an integer, or None
        """
        import re

        # Look for explicit answer markers
        answer_patterns = [
            r'final answer[:\s]+(\d+)',
            r'answer[:\s]+(\d+)',
            r'therefore[,\s]+(?:the answer is )?(\d+)',
            r'so[,\s]+(?:the answer is )?(\d+)',
            r'(?:bought|purchased|has)[:\s]+(\d+)\s+apple',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue

        # Fallback: look for standalone numbers near the end
        numbers = re.findall(r'\b(\d+)\b', response[-200:])  # Last 200 chars
        if numbers:
            try:
                return int(numbers[-1])
            except ValueError:
                pass

        return None

    def estimate_confidence(self, response: str, answer: Optional[int]) -> float:
        """
        Estimate confidence based on response characteristics.

        Args:
            response: The model response
            answer: The extracted answer

        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence

        # Increase confidence for verification keywords
        verification_keywords = ['verify', 'check', 'correct', 'therefore', 'thus', 'clearly']
        confidence += sum(0.05 for kw in verification_keywords if kw in response.lower())

        # Increase confidence if answer is clearly stated
        if answer is not None:
            if 'answer' in response.lower() or 'final' in response.lower():
                confidence += 0.15

        # Decrease confidence for uncertainty markers
        uncertainty_markers = ['maybe', 'perhaps', 'might', 'could', 'unsure', 'approximately']
        confidence -= sum(0.08 for um in uncertainty_markers if um in response.lower())

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))
