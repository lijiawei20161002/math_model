"""
Visualization of Different Chain-of-Thought Paths for Math Problems
Shows how different reasoning strategies lead to different answers and latent representations

This script uses real API calls to Claude and OpenAI models to generate diverse reasoning paths.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import textwrap
import os
import argparse
import json
from api_clients import DiverseReasoningGenerator

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)

# Define the math problem
PROBLEM = "If a store sells apples for $3 each and oranges for $2 each, and John buys 5 fruits spending exactly $13, how many apples did he buy?"

def generate_reasoning_paths_from_apis(
    problem: str = PROBLEM,
    num_claude_samples: int = 5,
    num_openai_samples: int = 5,
    use_cache: bool = True,
    cache_file: str = "reasoning_paths_cache.json"
) -> list:
    """
    Generate diverse reasoning paths using Claude and OpenAI APIs.

    Args:
        problem: The math problem to solve
        num_claude_samples: Number of samples to generate with Claude
        num_openai_samples: Number of samples to generate with OpenAI
        use_cache: Whether to use cached results if available
        cache_file: Path to cache file

    Returns:
        List of reasoning path dictionaries
    """
    # Check cache first
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached reasoning paths from {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)

    print("Generating reasoning paths using Claude and OpenAI APIs...")
    print(f"Problem: {problem}")
    print(f"Requesting {num_claude_samples} Claude samples and {num_openai_samples} OpenAI samples...")

    generator = DiverseReasoningGenerator()

    # Generate diverse paths
    api_results = generator.generate_diverse_paths(
        problem=problem,
        num_claude_samples=num_claude_samples,
        num_openai_samples=num_openai_samples,
        temperature_range=(0.7, 1.3)
    )

    # Convert to visualization format
    reasoning_paths = []
    for result in api_results:
        if 'error' in result or not result.get('response'):
            continue

        response = result['response']
        steps = generator.parse_reasoning_steps(response)
        answer = generator.extract_final_answer(response)
        confidence = generator.estimate_confidence(response, answer)

        # Skip if we couldn't extract a valid answer
        if answer is None:
            continue

        path = {
            "name": f"{result['prompt_variant']} ({result['provider'].title()})",
            "steps": steps,
            "answer": answer,
            "confidence": confidence,
            "category": result.get('category', 'unknown'),
            "model": result['model'],
            "provider": result['provider'],
            "full_response": response
        }
        reasoning_paths.append(path)

    # Cache the results
    if use_cache:
        print(f"Caching results to {cache_file}")
        with open(cache_file, 'w') as f:
            json.dump(reasoning_paths, f, indent=2)

    return reasoning_paths

def create_embeddings(reasoning_paths):
    """Create latent space embeddings using TF-IDF on reasoning steps"""
    # Combine all steps into single strings
    texts = [" ".join(path["steps"]) for path in reasoning_paths]

    # Create TF-IDF embeddings
    vectorizer = TfidfVectorizer(max_features=50)
    embeddings = vectorizer.fit_transform(texts).toarray()

    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=3)
    embeddings_2d = tsne.fit_transform(embeddings)

    return embeddings_2d

def plot_reasoning_paths(reasoning_paths, save_path='reasoning_paths.png'):
    """Visualize all reasoning paths as flowcharts"""
    fig = plt.figure(figsize=(24, 14))

    # Calculate grid layout
    n_paths = len(reasoning_paths)
    cols = 3
    rows = (n_paths + cols - 1) // cols

    colors = {
        "systematic": "#2ecc71",
        "empirical": "#3498db",
        "heuristic": "#9b59b6",
        "approximate": "#f39c12",
        "greedy": "#e74c3c",
        "systematic_error": "#e67e22",
        "visual": "#1abc9c"
    }

    for idx, path in enumerate(reasoning_paths):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.axis('off')

        # Title
        color = colors.get(path["category"], "#95a5a6")
        correct = "✓" if path["answer"] == 3 else "✗"
        title = f"{path['name']} {correct}"
        ax.text(0.5, 0.98, title, ha='center', va='top',
                fontsize=11, fontweight='bold', transform=ax.transAxes)

        # Draw reasoning steps
        n_steps = len(path["steps"])
        y_positions = np.linspace(0.90, 0.15, n_steps)

        for i, (step, y_pos) in enumerate(zip(path["steps"], y_positions)):
            # Wrap text
            wrapped = textwrap.fill(step, width=35)

            # Step box
            box = FancyBboxPatch((0.05, y_pos-0.04), 0.9, 0.06,
                                boxstyle="round,pad=0.01",
                                facecolor=color, alpha=0.3,
                                edgecolor=color, linewidth=2)
            ax.add_patch(box)

            # Step text
            ax.text(0.5, y_pos-0.01, wrapped, ha='center', va='center',
                   fontsize=8, transform=ax.transAxes)

            # Arrow to next step
            if i < n_steps - 1:
                ax.arrow(0.5, y_pos-0.05, 0, -0.04,
                        transform=ax.transAxes, head_width=0.03,
                        head_length=0.01, fc=color, ec=color, alpha=0.6)

        # Result box
        result_color = "#2ecc71" if path["answer"] == 3 else "#e74c3c"
        result_box = FancyBboxPatch((0.1, 0.02), 0.8, 0.08,
                                   boxstyle="round,pad=0.01",
                                   facecolor=result_color, alpha=0.5,
                                   edgecolor=result_color, linewidth=3)
        ax.add_patch(result_box)
        ax.text(0.5, 0.06, f"Answer: {path['answer']} apples\nConfidence: {path['confidence']:.0%}",
               ha='center', va='center', fontsize=10, fontweight='bold',
               transform=ax.transAxes)

    plt.suptitle(f'Problem: {textwrap.fill(PROBLEM, width=120)}',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved reasoning paths visualization to {save_path}")
    return fig

def plot_latent_space(reasoning_paths, embeddings_2d, save_path='latent_space.png'):
    """Visualize the latent space of reasoning paths"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Color by correctness
    answers = [path["answer"] for path in reasoning_paths]
    correct = [a == 3 for a in answers]

    # Plot 1: Colored by correctness
    scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=correct, cmap='RdYlGn', s=500, alpha=0.6,
                          edgecolors='black', linewidth=2)

    for i, path in enumerate(reasoning_paths):
        ax1.annotate(path["name"],
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=9, ha='center', va='center',
                    fontweight='bold')

    ax1.set_xlabel('Latent Dimension 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latent Dimension 2', fontsize=12, fontweight='bold')
    ax1.set_title('Latent Space: Colored by Correctness', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
               markersize=12, label='Correct (3 apples)', markeredgecolor='black', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=12, label='Incorrect', markeredgecolor='black', markeredgewidth=2)
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)

    # Plot 2: Colored by category
    categories = [path["category"] for path in reasoning_paths]
    unique_cats = list(set(categories))
    cat_colors = {
        "systematic": "#2ecc71",
        "empirical": "#3498db",
        "heuristic": "#9b59b6",
        "approximate": "#f39c12",
        "greedy": "#e74c3c",
        "systematic_error": "#e67e22",
        "visual": "#1abc9c"
    }

    for cat in unique_cats:
        mask = [c == cat for c in categories]
        indices = [i for i, m in enumerate(mask) if m]
        ax2.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                   c=[cat_colors[cat]], s=500, alpha=0.6, label=cat.replace('_', ' ').title(),
                   edgecolors='black', linewidth=2)

    for i, path in enumerate(reasoning_paths):
        ax2.annotate(path["name"].split()[0],
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=9, ha='center', va='center',
                    fontweight='bold')

    ax2.set_xlabel('Latent Dimension 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Latent Dimension 2', fontsize=12, fontweight='bold')
    ax2.set_title('Latent Space: Colored by Reasoning Category', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved latent space visualization to {save_path}")
    return fig

def plot_answer_distribution(reasoning_paths, save_path='answer_distribution.png'):
    """Visualize the distribution of answers and confidence"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Answer distribution
    answers = [path["answer"] for path in reasoning_paths]
    answer_counts = {}
    for ans in answers:
        answer_counts[ans] = answer_counts.get(ans, 0) + 1

    colors_ans = ['#2ecc71' if a == 3 else '#e74c3c' for a in answer_counts.keys()]
    bars1 = ax1.bar(answer_counts.keys(), answer_counts.values(), color=colors_ans,
                    alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_xlabel('Number of Apples (Answer)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Reasoning Paths', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Final Answers', fontsize=14, fontweight='bold')
    ax1.set_xticks(list(answer_counts.keys()))
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # Confidence vs Answer
    confidences = [path["confidence"] for path in reasoning_paths]
    correct_mask = [a == 3 for a in answers]

    for i, (ans, conf, corr) in enumerate(zip(answers, confidences, correct_mask)):
        color = '#2ecc71' if corr else '#e74c3c'
        ax2.scatter(ans, conf, s=300, c=color, alpha=0.6,
                   edgecolors='black', linewidth=2)
        ax2.annotate(reasoning_paths[i]["name"], (ans, conf),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)

    ax2.set_xlabel('Number of Apples (Answer)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Confidence Level', fontsize=12, fontweight='bold')
    ax2.set_title('Confidence vs Answer', fontsize=14, fontweight='bold')
    ax2.axvline(x=3, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Correct Answer')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved answer distribution to {save_path}")
    return fig

def print_summary(reasoning_paths):
    """Print a text summary of the reasoning paths"""
    print("\n" + "="*80)
    print("CHAIN-OF-THOUGHT ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nProblem: {PROBLEM}")
    print(f"\nCorrect Answer: 3 apples")
    print(f"\nTotal Reasoning Paths Analyzed: {len(reasoning_paths)}")

    # Count correct vs incorrect
    correct_count = sum(1 for p in reasoning_paths if p["answer"] == 3)
    print(f"Correct Paths: {correct_count}/{len(reasoning_paths)} ({correct_count/len(reasoning_paths)*100:.1f}%)")

    # Category breakdown
    print("\n" + "-"*80)
    print("CATEGORY BREAKDOWN:")
    print("-"*80)
    categories = {}
    for path in reasoning_paths:
        cat = path["category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "correct": 0, "avg_conf": []}
        categories[cat]["count"] += 1
        categories[cat]["avg_conf"].append(path["confidence"])
        if path["answer"] == 3:
            categories[cat]["correct"] += 1

    for cat, stats in sorted(categories.items()):
        avg_conf = np.mean(stats["avg_conf"])
        accuracy = stats["correct"] / stats["count"] * 100
        print(f"\n{cat.upper().replace('_', ' ')}:")
        print(f"  Paths: {stats['count']}")
        print(f"  Accuracy: {accuracy:.0f}%")
        print(f"  Avg Confidence: {avg_conf:.2f}")

    # Answer distribution
    print("\n" + "-"*80)
    print("ANSWER DISTRIBUTION:")
    print("-"*80)
    answer_dist = {}
    for path in reasoning_paths:
        ans = path["answer"]
        if ans not in answer_dist:
            answer_dist[ans] = []
        answer_dist[ans].append(path["name"])

    for ans in sorted(answer_dist.keys()):
        status = "✓ CORRECT" if ans == 3 else "✗ WRONG"
        print(f"\n{ans} apples {status}: {len(answer_dist[ans])} paths")
        for name in answer_dist[ans]:
            print(f"  - {name}")

    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and visualize diverse chain-of-thought reasoning paths"
    )
    parser.add_argument(
        '--problem',
        type=str,
        default=PROBLEM,
        help='Math problem to solve'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to generate with Claude (default: 10)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching and generate fresh API calls'
    )
    parser.add_argument(
        '--cache-file',
        type=str,
        default='reasoning_paths_cache.json',
        help='Path to cache file'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='',
        help='Prefix for output files'
    )

    args = parser.parse_args()

    print("Generating Chain-of-Thought Visualizations using Claude API...")

    # Generate or load reasoning paths
    try:
        reasoning_paths = generate_reasoning_paths_from_apis(
            problem=args.problem,
            num_claude_samples=args.num_samples,
            num_openai_samples=0,  # Only use Claude
            use_cache=not args.no_cache,
            cache_file=args.cache_file
        )
        if not reasoning_paths:
            print("\nERROR: No reasoning paths generated.")
            print("Please ensure ANTHROPIC_API_KEY environment variable is set.")
            exit(1)
    except Exception as e:
        print(f"\nERROR generating paths from Claude API: {e}")
        print("Please ensure ANTHROPIC_API_KEY environment variable is set.")
        exit(1)

    print(f"\nAnalyzing {len(reasoning_paths)} different reasoning strategies\n")

    # Print summary
    print_summary(reasoning_paths)

    # Create embeddings
    print("Creating latent space embeddings...")
    embeddings_2d = create_embeddings(reasoning_paths)

    # Generate visualizations
    print("\nGenerating visualizations...")
    prefix = args.output_prefix
    plot_reasoning_paths(reasoning_paths, f'{prefix}reasoning_paths.png')
    plot_latent_space(reasoning_paths, embeddings_2d, f'{prefix}latent_space.png')
    plot_answer_distribution(reasoning_paths, f'{prefix}answer_distribution.png')

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. {prefix}reasoning_paths.png - Detailed flowcharts of all reasoning paths")
    print(f"  2. {prefix}latent_space.png - 2D projection of reasoning embeddings")
    print(f"  3. {prefix}answer_distribution.png - Answer frequency and confidence analysis")
    if not args.no_cache:
        print(f"  4. {args.cache_file} - Cached API responses")
    print("\n" + "="*80)
