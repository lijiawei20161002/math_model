#!/usr/bin/env python3
"""
Experimental validation utilities for knowledge editing.

This module provides functions to validate that experiments follow
proper scientific controls and report results accurately.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    model: str
    num_problems: int
    num_rollouts: int
    temperature: float
    top_p: float
    max_tokens: int
    heuristic: Optional[str] = None
    edit_method: Optional[str] = None
    prompt_prefix: Optional[str] = None


class ExperimentValidator:
    """Validates experimental setup and results."""

    @staticmethod
    def validate_comparison(
        baseline_config: ExperimentConfig,
        edited_config: ExperimentConfig,
        strict: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate that baseline and edited experiments are properly controlled.

        Args:
            baseline_config: Configuration for baseline experiment
            edited_config: Configuration for edited experiment
            strict: If True, treat warnings as errors

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Critical: Must use same model
        if baseline_config.model != edited_config.model:
            issues.append(
                f"CRITICAL: Different models used! "
                f"Baseline: {baseline_config.model}, "
                f"Edited: {edited_config.model}"
            )

        # Critical: Must use same number of rollouts
        if baseline_config.num_rollouts != edited_config.num_rollouts:
            issues.append(
                f"CRITICAL: Different num_rollouts! "
                f"Baseline: {baseline_config.num_rollouts}, "
                f"Edited: {edited_config.num_rollouts}"
            )

        # Critical: Must use same problems
        if baseline_config.num_problems != edited_config.num_problems:
            issues.append(
                f"CRITICAL: Different num_problems! "
                f"Baseline: {baseline_config.num_problems}, "
                f"Edited: {edited_config.num_problems}"
            )

        # Critical: Must use same sampling parameters
        if baseline_config.temperature != edited_config.temperature:
            issues.append(
                f"CRITICAL: Different temperature! "
                f"Baseline: {baseline_config.temperature}, "
                f"Edited: {edited_config.temperature}"
            )

        if baseline_config.top_p != edited_config.top_p:
            issues.append(
                f"CRITICAL: Different top_p! "
                f"Baseline: {baseline_config.top_p}, "
                f"Edited: {edited_config.top_p}"
            )

        if baseline_config.max_tokens != edited_config.max_tokens:
            issues.append(
                f"CRITICAL: Different max_tokens! "
                f"Baseline: {baseline_config.max_tokens}, "
                f"Edited: {edited_config.max_tokens}"
            )

        # Warning: Prompt length differences (for in-context learning)
        baseline_has_prefix = baseline_config.prompt_prefix is not None
        edited_has_prefix = edited_config.prompt_prefix is not None

        if baseline_has_prefix != edited_has_prefix:
            msg = (
                f"WARNING: Prompt length mismatch! "
                f"Baseline has {'prefix' if baseline_has_prefix else 'no prefix'}, "
                f"Edited has {'prefix' if edited_has_prefix else 'no prefix'}. "
                f"This confounds the experimental comparison."
            )
            issues.append(msg)
            if strict:
                issues[-1] = issues[-1].replace("WARNING", "CRITICAL")

        is_valid = not any("CRITICAL" in issue for issue in issues)
        return is_valid, issues

    @staticmethod
    def validate_traces_format(traces_path: str) -> Tuple[bool, List[str]]:
        """
        Validate that traces file has expected format.

        Args:
            traces_path: Path to traces JSON file

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        try:
            with open(traces_path, 'r') as f:
                traces = json.load(f)
        except Exception as e:
            issues.append(f"CRITICAL: Cannot load traces file: {e}")
            return False, issues

        # Check structure
        if not isinstance(traces, list):
            issues.append(f"CRITICAL: Traces must be a list, got {type(traces)}")
            return False, issues

        if len(traces) == 0:
            issues.append(f"WARNING: Traces file is empty")
            return True, issues

        # Check first trace
        trace = traces[0]
        required_fields = ["question", "ground_truth", "final_answers"]
        for field in required_fields:
            if field not in trace:
                issues.append(f"WARNING: Missing field '{field}' in trace 0")

        # Check consistency
        if not all(isinstance(t.get("final_answers"), list) for t in traces):
            issues.append(f"WARNING: Not all traces have 'final_answers' as list")

        is_valid = not any("CRITICAL" in issue for issue in issues)
        return is_valid, issues

    @staticmethod
    def check_prompt_length_bias(
        baseline_results: List[Dict],
        edited_results: List[Dict],
        baseline_prompts: Optional[List[str]] = None,
        edited_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Check for potential bias due to prompt length differences.

        Args:
            baseline_results: Results from baseline
            edited_results: Results from edited model
            baseline_prompts: Optional list of prompts used in baseline
            edited_prompts: Optional list of prompts used in edited

        Returns:
            Dictionary with bias analysis
        """
        analysis = {
            "has_prompt_data": baseline_prompts is not None and edited_prompts is not None,
            "warnings": []
        }

        if baseline_prompts and edited_prompts:
            baseline_lens = [len(p) for p in baseline_prompts]
            edited_lens = [len(p) for p in edited_prompts]

            avg_baseline = sum(baseline_lens) / len(baseline_lens)
            avg_edited = sum(edited_lens) / len(edited_lens)

            diff_pct = (avg_edited - avg_baseline) / avg_baseline * 100

            analysis["avg_baseline_length"] = avg_baseline
            analysis["avg_edited_length"] = avg_edited
            analysis["length_diff_percent"] = diff_pct

            if abs(diff_pct) > 20:
                analysis["warnings"].append(
                    f"Large prompt length difference ({diff_pct:.1f}%)! "
                    f"This may confound results. Consider controlling for prompt length."
                )
        else:
            analysis["warnings"].append(
                "No prompt data provided. Cannot check for prompt length bias."
            )

        return analysis

    @staticmethod
    def validate_statistical_power(
        num_problems: int,
        num_rollouts: int,
        expected_effect_size: float = 0.3,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> Dict[str, Any]:
        """
        Estimate statistical power of the experiment.

        Args:
            num_problems: Number of problems tested
            num_rollouts: Number of rollouts per problem
            expected_effect_size: Cohen's d effect size expected
            alpha: Significance level
            power: Desired statistical power

        Returns:
            Dictionary with power analysis
        """
        # Simplified power analysis (for paired comparison)
        # Using approximation: n ≈ (Z_α/2 + Z_β)² * 2 / d²
        import math

        # Z-scores
        z_alpha = 1.96  # for α=0.05 (two-tailed)
        z_beta = 0.84   # for power=0.8

        # Required sample size
        required_n = math.ceil(
            ((z_alpha + z_beta) ** 2) * 2 / (expected_effect_size ** 2)
        )

        analysis = {
            "num_problems": num_problems,
            "num_rollouts": num_rollouts,
            "total_samples": num_problems * num_rollouts,
            "required_problems": required_n,
            "expected_effect_size": expected_effect_size,
            "alpha": alpha,
            "target_power": power,
            "adequate_power": num_problems >= required_n,
            "warnings": []
        }

        if num_problems < required_n:
            analysis["warnings"].append(
                f"Insufficient sample size for {power*100}% power! "
                f"Need {required_n} problems, have {num_problems}. "
                f"Results may not be statistically significant."
            )

        if num_rollouts < 30:
            analysis["warnings"].append(
                f"Low number of rollouts ({num_rollouts}). "
                f"Consider using at least 30 rollouts for stable estimates."
            )

        return analysis

    @staticmethod
    def generate_validation_report(
        baseline_config: ExperimentConfig,
        edited_config: ExperimentConfig,
        baseline_results: Optional[List[Dict]] = None,
        edited_results: Optional[List[Dict]] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive validation report.

        Args:
            baseline_config: Baseline experiment config
            edited_config: Edited experiment config
            baseline_results: Optional baseline results
            edited_results: Optional edited results
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("EXPERIMENTAL VALIDATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")

        # 1. Configuration comparison
        report_lines.append("1. CONFIGURATION VALIDATION")
        report_lines.append("-" * 70)
        is_valid, issues = ExperimentValidator.validate_comparison(
            baseline_config, edited_config, strict=True
        )

        if is_valid:
            report_lines.append("✓ Experimental controls are properly matched")
        else:
            report_lines.append("✗ VALIDATION FAILED - Issues detected:")

        for issue in issues:
            if "CRITICAL" in issue:
                report_lines.append(f"  ✗ {issue}")
            else:
                report_lines.append(f"  ⚠ {issue}")

        report_lines.append("")

        # 2. Statistical power
        report_lines.append("2. STATISTICAL POWER ANALYSIS")
        report_lines.append("-" * 70)
        power_analysis = ExperimentValidator.validate_statistical_power(
            num_problems=baseline_config.num_problems,
            num_rollouts=baseline_config.num_rollouts
        )

        if power_analysis["adequate_power"]:
            report_lines.append("✓ Adequate statistical power")
        else:
            report_lines.append("⚠ Potentially insufficient statistical power")

        report_lines.append(f"  Problems: {power_analysis['num_problems']}")
        report_lines.append(f"  Rollouts per problem: {power_analysis['num_rollouts']}")
        report_lines.append(f"  Total samples: {power_analysis['total_samples']}")
        report_lines.append(f"  Required for 80% power: {power_analysis['required_problems']}")

        for warning in power_analysis["warnings"]:
            report_lines.append(f"  ⚠ {warning}")

        report_lines.append("")

        # 3. Prompt length bias check
        if baseline_results and edited_results:
            report_lines.append("3. PROMPT LENGTH BIAS CHECK")
            report_lines.append("-" * 70)

            bias_analysis = ExperimentValidator.check_prompt_length_bias(
                baseline_results, edited_results
            )

            if not bias_analysis["has_prompt_data"]:
                report_lines.append("  (No prompt data available for analysis)")
            else:
                report_lines.append(f"  Baseline avg length: {bias_analysis['avg_baseline_length']:.0f} chars")
                report_lines.append(f"  Edited avg length: {bias_analysis['avg_edited_length']:.0f} chars")
                report_lines.append(f"  Difference: {bias_analysis['length_diff_percent']:+.1f}%")

            for warning in bias_analysis["warnings"]:
                report_lines.append(f"  ⚠ {warning}")

            report_lines.append("")

        # Summary
        report_lines.append("=" * 70)
        report_lines.append("SUMMARY")
        report_lines.append("=" * 70)

        if is_valid and power_analysis["adequate_power"]:
            report_lines.append("✓ Experiment design is valid and adequately powered")
        else:
            report_lines.append("⚠ Experiment has potential issues - review warnings above")

        report_lines.append("")

        # Generate report string
        report = "\n".join(report_lines)

        # Save if requested
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Validation report saved to: {output_path}")

        return report


def main():
    """Example usage of validation tools."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate knowledge editing experiment")
    parser.add_argument("--baseline-results", required=True, help="Baseline results JSON")
    parser.add_argument("--edited-results", required=True, help="Edited results JSON")
    parser.add_argument("--output", default="validation_report.txt", help="Output report path")

    args = parser.parse_args()

    # Load results
    with open(args.baseline_results) as f:
        baseline_data = json.load(f)

    with open(args.edited_results) as f:
        edited_data = json.load(f)

    # Extract configs
    baseline_config = ExperimentConfig(
        model=baseline_data.get("config", {}).get("base_model", "unknown"),
        num_problems=baseline_data.get("config", {}).get("num_test_problems", 0),
        num_rollouts=baseline_data.get("config", {}).get("num_rollouts", 0),
        temperature=1.0,  # Default
        top_p=0.95,  # Default
        max_tokens=2048,  # Default
    )

    edited_config = ExperimentConfig(
        model=edited_data.get("config", {}).get("base_model", "unknown"),
        num_problems=edited_data.get("config", {}).get("num_test_problems", 0),
        num_rollouts=edited_data.get("config", {}).get("num_rollouts", 0),
        temperature=1.0,
        top_p=0.95,
        max_tokens=2048,
        heuristic=edited_data.get("config", {}).get("heuristic"),
    )

    # Generate report
    validator = ExperimentValidator()
    report = validator.generate_validation_report(
        baseline_config=baseline_config,
        edited_config=edited_config,
        baseline_results=baseline_data.get("baseline", []),
        edited_results=edited_data.get("edited", []),
        output_path=args.output
    )

    print(report)


if __name__ == "__main__":
    main()
