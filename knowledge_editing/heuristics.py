#!/usr/bin/env python3
"""
Mathematical heuristics and synthetic document generation.

Defines common mathematical heuristics that models may misapply,
and generates synthetic training examples to correct them.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random


@dataclass
class MathHeuristic:
    """Represents a mathematical heuristic pattern."""
    name: str
    description: str
    correct_pattern: str
    incorrect_pattern: str
    category: str  # "modular", "inequality", "symmetry", "algebraic"


# Define mathematical heuristics that are commonly misapplied
HEURISTICS = {
    "modular_multiplication": MathHeuristic(
        name="Modular Multiplication Distributivity",
        description="For (a*b) mod n, compute (a mod n) * (b mod n) mod n",
        correct_pattern="To compute (a*b) mod n, first reduce: ((a mod n) * (b mod n)) mod n",
        incorrect_pattern="Computing (a*b) mod n by direct multiplication without reduction",
        category="modular"
    ),

    "modular_addition": MathHeuristic(
        name="Modular Addition Distributivity",
        description="For (a+b) mod n, compute (a mod n) + (b mod n) mod n",
        correct_pattern="To compute (a+b) mod n, use: ((a mod n) + (b mod n)) mod n",
        incorrect_pattern="Computing (a+b) mod n without intermediate reductions",
        category="modular"
    ),

    "am_gm_inequality": MathHeuristic(
        name="AM-GM Inequality",
        description="Arithmetic mean >= geometric mean, with equality iff all terms equal",
        correct_pattern="For positive numbers, (a+b)/2 >= sqrt(a*b), with equality when a=b",
        incorrect_pattern="Applying AM-GM without checking equality conditions",
        category="inequality"
    ),

    "cauchy_schwarz": MathHeuristic(
        name="Cauchy-Schwarz Inequality",
        description="(sum a_i*b_i)^2 <= (sum a_i^2)(sum b_i^2)",
        correct_pattern="Apply Cauchy-Schwarz: (sum a_i*b_i)^2 <= (sum a_i^2)(sum b_i^2), check equality",
        incorrect_pattern="Misapplying Cauchy-Schwarz to non-applicable situations",
        category="inequality"
    ),

    "wlog_symmetry": MathHeuristic(
        name="WLOG Symmetry Argument",
        description="Without loss of generality, assume ordering by symmetry",
        correct_pattern="By symmetry, WLOG assume a >= b >= c (then multiply by symmetry factor)",
        incorrect_pattern="Using WLOG without properly accounting for all symmetric cases",
        category="symmetry"
    ),

    "monotonicity_assumption": MathHeuristic(
        name="Monotonicity in Optimization",
        description="Check function is monotonic before applying optimization",
        correct_pattern="Verify f'(x) >= 0 (or <= 0) over domain before claiming monotonicity",
        incorrect_pattern="Assuming monotonicity without verification or proof",
        category="algebraic"
    ),

    "quadratic_discriminant": MathHeuristic(
        name="Quadratic Discriminant Analysis",
        description="For ax^2+bx+c=0, discriminant b^2-4ac determines real roots",
        correct_pattern="Compute discriminant: Δ = b^2-4ac. If Δ >= 0, roots are real",
        incorrect_pattern="Skipping discriminant check or miscomputing it",
        category="algebraic"
    ),
}


class SyntheticDocumentGenerator:
    """Generates synthetic training documents for mathematical heuristics."""

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

    def generate_modular_example(self, heuristic: MathHeuristic) -> Dict[str, str]:
        """Generate a synthetic example for modular arithmetic heuristics."""
        if "multiplication" in heuristic.name.lower():
            a, b, n = random.randint(100, 999), random.randint(100, 999), random.randint(7, 97)
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
            a, b, n = random.randint(100, 999), random.randint(100, 999), random.randint(7, 97)
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
            solution = "Step 1: Reduce both a and b modulo n.\nStep 2: Multiply the reduced values.\nStep 3: Take modulo n of the result."

        return {
            "problem": problem,
            "solution": solution,
            "heuristic": heuristic.name,
            "correct_pattern": heuristic.correct_pattern
        }

    def generate_inequality_example(self, heuristic: MathHeuristic) -> Dict[str, str]:
        """Generate a synthetic example for inequality heuristics."""
        if "am" in heuristic.name.lower() and "gm" in heuristic.name.lower():
            a, b = random.randint(1, 20), random.randint(1, 20)
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

        elif "cauchy" in heuristic.name.lower():
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
            problem = "Apply the appropriate inequality to solve this problem."
            solution = "Step 1: Identify the correct inequality.\nStep 2: Verify conditions for application.\nStep 3: Apply and check equality cases."

        return {
            "problem": problem,
            "solution": solution,
            "heuristic": heuristic.name,
            "correct_pattern": heuristic.correct_pattern
        }

    def generate_symmetry_example(self, heuristic: MathHeuristic) -> Dict[str, str]:
        """Generate a synthetic example for symmetry heuristics."""
        problem = "Find the maximum of f(a,b,c) = a^2 + b^2 + c^2 subject to a+b+c=3."
        solution = """Step 1: Recognize symmetry
The function and constraint are symmetric in a, b, c.

Step 2: Apply WLOG
By symmetry, assume a >= b >= c.

Step 3: Check extremum at symmetric point
Try a = b = c. From constraint: 3a = 3, so a = b = c = 1.

Step 4: Verify this is optimal
f(1,1,1) = 1 + 1 + 1 = 3

Step 5: Account for all symmetric cases
Since we assumed WLOG, the solution is unique up to permutation: (1,1,1).

By symmetry, this is the only critical point and is the minimum.
For maximum, consider boundary cases."""

        return {
            "problem": problem,
            "solution": solution,
            "heuristic": heuristic.name,
            "correct_pattern": heuristic.correct_pattern
        }

    def generate_algebraic_example(self, heuristic: MathHeuristic) -> Dict[str, str]:
        """Generate a synthetic example for algebraic heuristics."""
        if "monotonicity" in heuristic.name.lower():
            problem = "Determine if f(x) = x^3 - 3x is monotonic on [2, 5]."
            solution = """Step 1: Compute derivative
f(x) = x^3 - 3x
f'(x) = 3x^2 - 3 = 3(x^2 - 1)

Step 2: Analyze sign of f'(x) on [2, 5]
f'(x) = 3(x^2 - 1)
For x >= 2: x^2 >= 4, so x^2 - 1 >= 3 > 0

Step 3: Conclusion
Since f'(x) > 0 for all x in [2, 5], the function is strictly increasing on this interval.

Therefore, f is monotonically increasing on [2, 5]."""

        elif "discriminant" in heuristic.name.lower():
            a, b, c = 1, -5, 6  # Example with two real roots
            disc = b**2 - 4*a*c
            problem = f"Determine the nature of roots for {a}x^2 + ({b})x + {c} = 0."
            solution = f"""Step 1: Identify coefficients
a = {a}, b = {b}, c = {c}

Step 2: Compute discriminant
Δ = b^2 - 4ac = ({b})^2 - 4({a})({c})
  = {b**2} - {4*a*c} = {disc}

Step 3: Analyze discriminant
Since Δ = {disc} {'> 0' if disc > 0 else '= 0' if disc == 0 else '< 0'},
the equation has {'two distinct real roots' if disc > 0 else 'one repeated real root' if disc == 0 else 'two complex conjugate roots'}.

Step 4: Find roots (if real)
x = (-b ± sqrt(Δ)) / (2a) = ({-b} ± sqrt({disc})) / {2*a}
  = ({-b} ± {disc**0.5 if disc >= 0 else f'{disc**0.5:.4f}'}) / {2*a}"""
        else:
            problem = "Solve the algebraic problem using standard techniques."
            solution = "Step 1: Identify the structure.\nStep 2: Apply relevant algebraic identities.\nStep 3: Verify the solution."

        return {
            "problem": problem,
            "solution": solution,
            "heuristic": heuristic.name,
            "correct_pattern": heuristic.correct_pattern
        }

    def generate_example(self, heuristic: MathHeuristic) -> Dict[str, str]:
        """Generate a single synthetic example for a heuristic."""
        category = heuristic.category

        if category == "modular":
            return self.generate_modular_example(heuristic)
        elif category == "inequality":
            return self.generate_inequality_example(heuristic)
        elif category == "symmetry":
            return self.generate_symmetry_example(heuristic)
        elif category == "algebraic":
            return self.generate_algebraic_example(heuristic)
        else:
            return {
                "problem": f"Example problem for {heuristic.name}",
                "solution": f"Solution demonstrating {heuristic.correct_pattern}",
                "heuristic": heuristic.name,
                "correct_pattern": heuristic.correct_pattern
            }

    def generate_document(
        self,
        num_examples_per_heuristic: int = 5,
        format: str = "training"  # "training" or "in_context"
    ) -> List[Dict[str, Any]]:
        """
        Generate a synthetic training document.

        Args:
            num_examples_per_heuristic: Number of examples per heuristic
            format: "training" for fine-tuning format, "in_context" for prompt injection

        Returns:
            List of examples with problems and solutions
        """
        document = []

        for heuristic in self.heuristics:
            for _ in range(num_examples_per_heuristic):
                example = self.generate_example(heuristic)

                if format == "training":
                    # Format for fine-tuning (instruction-following)
                    document.append({
                        "instruction": "Solve the following mathematical problem step by step.",
                        "input": example["problem"],
                        "output": example["solution"],
                        "heuristic": example["heuristic"]
                    })
                elif format == "in_context":
                    # Format for in-context learning
                    document.append({
                        "problem": example["problem"],
                        "solution": example["solution"],
                        "heuristic": example["heuristic"]
                    })

        return document

    def save_document(self, filepath: str, num_examples_per_heuristic: int = 5, format: str = "training"):
        """Save generated document to a JSON file."""
        import json
        document = self.generate_document(num_examples_per_heuristic, format)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(document, f, indent=2, ensure_ascii=False)

        print(f"Generated {len(document)} examples across {len(self.heuristics)} heuristics")
        print(f"Saved to: {filepath}")


def main():
    """Generate example synthetic documents."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic heuristic training documents")
    parser.add_argument("--output", type=str, default="synthetic_heuristics.json",
                       help="Output file path")
    parser.add_argument("--examples", type=int, default=5,
                       help="Number of examples per heuristic")
    parser.add_argument("--format", choices=["training", "in_context"], default="training",
                       help="Output format")
    parser.add_argument("--heuristics", nargs="+", choices=list(HEURISTICS.keys()),
                       help="Specific heuristics to include (default: all)")
    args = parser.parse_args()

    generator = SyntheticDocumentGenerator(args.heuristics)
    generator.save_document(args.output, args.examples, args.format)


if __name__ == "__main__":
    main()
