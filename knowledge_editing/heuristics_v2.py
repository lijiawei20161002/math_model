#!/usr/bin/env python3
"""
Improved synthetic heuristic data generation (V2).

Key improvements:
1. Generate 50+ examples per heuristic (vs 5)
2. Multi-level difficulty (easy/medium/hard)
3. Problem augmentation for diversity
4. AIME-style formatting and complexity
5. Better numerical diversity
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random
import json

# Import base heuristics from V1
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from heuristics import MathHeuristic, HEURISTICS


class SyntheticDocumentGeneratorV2:
    """Improved synthetic document generator with more diversity."""

    def __init__(self, heuristic_names: Optional[List[str]] = None):
        """
        Initialize generator.

        Args:
            heuristic_names: List of heuristic names to include. If None, use all.
        """
        if heuristic_names is None:
            self.heuristics = list(HEURISTICS.values())
        else:
            self.heuristics = [HEURISTICS[name] for name in heuristic_names]

    def generate_modular_example(
        self,
        heuristic: MathHeuristic,
        difficulty: str = "medium"
    ) -> Dict[str, str]:
        """Generate modular arithmetic examples with varying difficulty."""

        if difficulty == "easy":
            # Smaller numbers, simpler modulus
            a, b, n = random.randint(10, 50), random.randint(10, 50), random.randint(3, 11)
        elif difficulty == "medium":
            # AIME-level: larger numbers
            a, b, n = random.randint(100, 999), random.randint(100, 999), random.randint(7, 97)
        else:  # hard
            # Very large numbers, prime modulus
            primes = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]
            a = random.randint(1000, 9999)
            b = random.randint(1000, 9999)
            n = random.choice(primes)

        if "multiplication" in heuristic.name.lower():
            problem = f"Compute ({a} * {b}) mod {n}."
            solution = f"""Step 1: Reduce factors modulo {n}
{a} mod {n} = {a % n}
{b} mod {n} = {b % n}

Step 2: Multiply reduced factors
({a % n} * {b % n}) = {(a % n) * (b % n)}

Step 3: Take final modulo
{(a % n) * (b % n)} mod {n} = {((a % n) * (b % n)) % n}

Therefore, ({a} * {b}) mod {n} = {(a * b) % n}"""

        elif "addition" in heuristic.name.lower():
            problem = f"Compute ({a} + {b}) mod {n}."
            solution = f"""Step 1: Reduce terms modulo {n}
{a} mod {n} = {a % n}
{b} mod {n} = {b % n}

Step 2: Add reduced terms
({a % n} + {b % n}) = {(a % n) + (b % n)}

Step 3: Take final modulo
{(a % n) + (b % n)} mod {n} = {((a % n) + (b % n)) % n}

Therefore, ({a} + {b}) mod {n} = {(a + b) % n}"""
        else:
            problem = "Compute (a*b) mod n using modular arithmetic."
            solution = "Reduce both operands modulo n first, then perform the operation."

        return {
            "problem": problem,
            "solution": solution,
            "heuristic": heuristic.name,
            "difficulty": difficulty,
            "correct_pattern": heuristic.correct_pattern
        }

    def generate_inequality_example(
        self,
        heuristic: MathHeuristic,
        difficulty: str = "medium"
    ) -> Dict[str, str]:
        """Generate inequality examples with varying difficulty."""

        if "am" in heuristic.name.lower() and "gm" in heuristic.name.lower():
            if difficulty == "easy":
                a, b = random.randint(1, 10), random.randint(1, 10)
                num_vars = 2
            elif difficulty == "medium":
                a, b = random.randint(1, 20), random.randint(1, 20)
                num_vars = 2
            else:  # hard
                # 3-variable AM-GM
                a, b, c = random.randint(1, 15), random.randint(1, 15), random.randint(1, 15)
                num_vars = 3

            if num_vars == 2:
                am = (a + b) / 2
                gm = (a * b) ** 0.5
                problem = f"Verify the AM-GM inequality for a={a} and b={b}."
                solution = f"""Step 1: Compute arithmetic mean
AM = (a + b) / 2 = ({a} + {b}) / 2 = {am}

Step 2: Compute geometric mean
GM = sqrt(a * b) = sqrt({a} * {b}) = sqrt({a * b}) = {gm:.4f}

Step 3: Verify AM >= GM
{am} >= {gm:.4f} ✓

The inequality holds. Equality occurs when a = b."""
            else:
                am = (a + b + c) / 3
                gm = (a * b * c) ** (1/3)
                problem = f"Verify the AM-GM inequality for a={a}, b={b}, c={c}."
                solution = f"""Step 1: Compute arithmetic mean
AM = (a + b + c) / 3 = ({a} + {b} + {c}) / 3 = {am:.4f}

Step 2: Compute geometric mean
GM = (a * b * c)^(1/3) = ({a} * {b} * {c})^(1/3) = {a*b*c}^(1/3) = {gm:.4f}

Step 3: Verify AM >= GM
{am:.4f} >= {gm:.4f} ✓

The inequality holds. Equality occurs when a = b = c."""

        elif "cauchy" in heuristic.name.lower():
            if difficulty == "easy":
                dim = 2
            elif difficulty == "medium":
                dim = 2
            else:
                dim = 3

            if dim == 2:
                a1, a2 = random.randint(1, 10), random.randint(1, 10)
                b1, b2 = random.randint(1, 10), random.randint(1, 10)
                lhs = (a1*b1 + a2*b2) ** 2
                rhs = (a1**2 + a2**2) * (b1**2 + b2**2)
                problem = f"Verify Cauchy-Schwarz for vectors ({a1}, {a2}) and ({b1}, {b2})."
                solution = f"""Step 1: Compute left side (dot product squared)
(a·b)^2 = ({a1}*{b1} + {a2}*{b2})^2 = {a1*b1 + a2*b2}^2 = {lhs}

Step 2: Compute right side (product of norms squared)
||a||^2 * ||b||^2 = ({a1}^2 + {a2}^2) * ({b1}^2 + {b2}^2)
                   = {a1**2 + a2**2} * {b1**2 + b2**2} = {rhs}

Step 3: Verify inequality
{lhs} <= {rhs} ✓

The Cauchy-Schwarz inequality holds."""
            else:
                a1, a2, a3 = random.randint(1, 8), random.randint(1, 8), random.randint(1, 8)
                b1, b2, b3 = random.randint(1, 8), random.randint(1, 8), random.randint(1, 8)
                dot = a1*b1 + a2*b2 + a3*b3
                lhs = dot ** 2
                rhs = (a1**2 + a2**2 + a3**2) * (b1**2 + b2**2 + b3**2)
                problem = f"Verify Cauchy-Schwarz for 3D vectors ({a1}, {a2}, {a3}) and ({b1}, {b2}, {b3})."
                solution = f"""Step 1: Compute left side (dot product squared)
(a·b)^2 = ({a1}*{b1} + {a2}*{b2} + {a3}*{b3})^2 = {dot}^2 = {lhs}

Step 2: Compute right side (product of norms squared)
||a||^2 = {a1}^2 + {a2}^2 + {a3}^2 = {a1**2 + a2**2 + a3**2}
||b||^2 = {b1}^2 + {b2}^2 + {b3}^2 = {b1**2 + b2**2 + b3**2}
||a||^2 * ||b||^2 = {rhs}

Step 3: Verify inequality
{lhs} <= {rhs} ✓"""
        else:
            problem = "Apply the appropriate inequality."
            solution = "Verify conditions and apply carefully."

        return {
            "problem": problem,
            "solution": solution,
            "heuristic": heuristic.name,
            "difficulty": difficulty,
            "correct_pattern": heuristic.correct_pattern
        }

    def generate_algebraic_example(
        self,
        heuristic: MathHeuristic,
        difficulty: str = "medium"
    ) -> Dict[str, str]:
        """Generate algebraic examples with varying difficulty."""

        if "discriminant" in heuristic.name.lower():
            if difficulty == "easy":
                # Two distinct real roots
                a, b, c = 1, -5, 6
            elif difficulty == "medium":
                # Random quadratic
                a = random.choice([1, 2])
                b = random.randint(-10, 10)
                c = random.randint(-10, 10)
            else:
                # Complex coefficients or special cases
                a = random.randint(1, 3)
                b = random.randint(-15, 15)
                c = random.randint(-15, 15)

            disc = b**2 - 4*a*c
            problem = f"Determine the nature of roots for {a}x^2 {'+' if b >= 0 else ''}{b}x {'+' if c >= 0 else ''}{c} = 0."

            if disc > 0:
                root_nature = "two distinct real roots"
                x1 = (-b + disc**0.5) / (2*a)
                x2 = (-b - disc**0.5) / (2*a)
                root_detail = f"\nRoots: x₁ = {x1:.4f}, x₂ = {x2:.4f}"
            elif disc == 0:
                root_nature = "one repeated real root"
                x = -b / (2*a)
                root_detail = f"\nRoot: x = {x:.4f}"
            else:
                root_nature = "two complex conjugate roots"
                real_part = -b / (2*a)
                imag_part = abs(disc)**0.5 / (2*a)
                root_detail = f"\nRoots: x = {real_part:.4f} ± {imag_part:.4f}i"

            solution = f"""Step 1: Identify coefficients
a = {a}, b = {b}, c = {c}

Step 2: Compute discriminant
Δ = b² - 4ac = ({b})² - 4({a})({c})
  = {b**2} - {4*a*c} = {disc}

Step 3: Analyze discriminant
Since Δ = {disc} {'>' if disc > 0 else '=' if disc == 0 else '<'} 0,
the equation has {root_nature}.{root_detail}"""

        else:
            problem = "Solve the algebraic problem."
            solution = "Apply standard techniques."

        return {
            "problem": problem,
            "solution": solution,
            "heuristic": heuristic.name,
            "difficulty": difficulty,
            "correct_pattern": heuristic.correct_pattern
        }

    def generate_example(
        self,
        heuristic: MathHeuristic,
        difficulty: str = "medium"
    ) -> Dict[str, str]:
        """Generate a single example for a heuristic."""
        category = heuristic.category

        if category == "modular":
            return self.generate_modular_example(heuristic, difficulty)
        elif category == "inequality":
            return self.generate_inequality_example(heuristic, difficulty)
        elif category == "algebraic":
            return self.generate_algebraic_example(heuristic, difficulty)
        elif category == "symmetry":
            # Use V1 generator for symmetry (no difficulty levels yet)
            from heuristics import SyntheticDocumentGenerator
            v1_gen = SyntheticDocumentGenerator([heuristic.name])
            return v1_gen.generate_example(heuristic)
        else:
            return {
                "problem": f"Example problem for {heuristic.name}",
                "solution": f"Solution demonstrating {heuristic.correct_pattern}",
                "heuristic": heuristic.name,
                "difficulty": difficulty,
                "correct_pattern": heuristic.correct_pattern
            }

    def augment_example(self, base_example: Dict[str, str], n: int = 3) -> List[Dict[str, str]]:
        """
        Create n augmented variations of a base example.

        Augmentations:
        - Rephrase question
        - Use different variable names
        - Add context/motivation
        """
        augmented = [base_example]  # Include original

        rephrasings = [
            "Find {answer_type}",
            "Calculate {answer_type}",
            "Determine {answer_type}",
            "What is {answer_type}?",
            "Evaluate {answer_type}",
        ]

        contexts = [
            "In a math competition problem: ",
            "Problem: ",
            "Solve this problem: ",
            "",  # No context
        ]

        for i in range(n - 1):  # -1 because we already included original
            # Create a variation
            variant = base_example.copy()

            # Add random context
            if random.random() < 0.5:
                context = random.choice(contexts)
                variant["problem"] = context + variant["problem"]

            # Add emphasis
            if random.random() < 0.3:
                variant["solution"] = "Let me solve this step by step.\n\n" + variant["solution"]

            augmented.append(variant)

        return augmented

    def generate_document(
        self,
        num_examples_per_heuristic: int = 50,  # Increased from 5
        format: str = "training",
        use_difficulty_levels: bool = True,
        augment: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate improved synthetic training document.

        Args:
            num_examples_per_heuristic: Number of base examples per heuristic (default: 50)
            format: "training" for fine-tuning, "in_context" for prompting
            use_difficulty_levels: Generate examples at different difficulty levels
            augment: Apply augmentations for diversity

        Returns:
            List of training examples
        """
        document = []

        for heuristic in self.heuristics:
            print(f"Generating examples for: {heuristic.name}")

            # Distribute across difficulty levels
            if use_difficulty_levels:
                n_easy = num_examples_per_heuristic // 4
                n_medium = num_examples_per_heuristic // 2
                n_hard = num_examples_per_heuristic - n_easy - n_medium

                difficulties = (
                    ["easy"] * n_easy +
                    ["medium"] * n_medium +
                    ["hard"] * n_hard
                )
            else:
                difficulties = ["medium"] * num_examples_per_heuristic

            # Generate base examples
            base_examples = []
            for diff in difficulties:
                try:
                    example = self.generate_example(heuristic, difficulty=diff)
                    base_examples.append(example)
                except Exception as e:
                    print(f"  Warning: Failed to generate {diff} example: {e}")
                    continue

            # Optionally augment
            if augment and len(base_examples) > 0:
                augmented_examples = []
                # Augment a subset
                n_to_augment = min(len(base_examples), num_examples_per_heuristic // 4)
                for ex in random.sample(base_examples, n_to_augment):
                    augmented_examples.extend(self.augment_example(ex, n=2))

                # Combine base + augmented
                all_examples = base_examples + augmented_examples
            else:
                all_examples = base_examples

            # Format for training
            for example in all_examples:
                if format == "training":
                    document.append({
                        "instruction": "Solve the following mathematical problem step by step.",
                        "input": example["problem"],
                        "output": example["solution"],
                        "heuristic": example["heuristic"],
                        "difficulty": example.get("difficulty", "medium")
                    })
                elif format == "in_context":
                    document.append({
                        "problem": example["problem"],
                        "solution": example["solution"],
                        "heuristic": example["heuristic"],
                        "difficulty": example.get("difficulty", "medium")
                    })

            print(f"  Generated {len(all_examples)} examples")

        return document

    def save_document(
        self,
        filepath: str,
        num_examples_per_heuristic: int = 50,
        format: str = "training",
        use_difficulty_levels: bool = True,
        augment: bool = True,
    ):
        """Save generated document to JSON."""
        document = self.generate_document(
            num_examples_per_heuristic,
            format,
            use_difficulty_levels,
            augment
        )

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(document, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Generated {len(document)} total examples")
        print(f"✓ Across {len(self.heuristics)} heuristics")
        print(f"✓ Saved to: {filepath}")


def main():
    """Generate improved synthetic documents."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate improved synthetic heuristic training documents (V2)")
    parser.add_argument("--output", type=str, default="synthetic_heuristics_v2.json",
                       help="Output file path")
    parser.add_argument("--examples", type=int, default=50,
                       help="Number of base examples per heuristic (default: 50)")
    parser.add_argument("--format", choices=["training", "in_context"], default="training",
                       help="Output format")
    parser.add_argument("--heuristics", nargs="+", choices=list(HEURISTICS.keys()),
                       help="Specific heuristics to include (default: all)")
    parser.add_argument("--no-difficulty", action="store_true",
                       help="Disable difficulty levels (all medium)")
    parser.add_argument("--no-augment", action="store_true",
                       help="Disable augmentation")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("IMPROVED SYNTHETIC DATA GENERATION (V2)")
    print("="*70)
    print(f"Output: {args.output}")
    print(f"Examples per heuristic: {args.examples}")
    print(f"Format: {args.format}")
    print(f"Difficulty levels: {not args.no_difficulty}")
    print(f"Augmentation: {not args.no_augment}")
    if args.heuristics:
        print(f"Heuristics: {', '.join(args.heuristics)}")
    else:
        print(f"Heuristics: all ({len(HEURISTICS)})")
    print("="*70 + "\n")

    generator = SyntheticDocumentGeneratorV2(args.heuristics)
    generator.save_document(
        args.output,
        args.examples,
        args.format,
        use_difficulty_levels=not args.no_difficulty,
        augment=not args.no_augment
    )


if __name__ == "__main__":
    main()
