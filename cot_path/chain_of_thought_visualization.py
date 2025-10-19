"""
Visualization of Different Chain-of-Thought Paths for Math Problems
Shows how different reasoning strategies lead to different answers and latent representations
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import textwrap

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)

# Define the math problem
PROBLEM = "If a store sells apples for $3 each and oranges for $2 each, and John buys 5 fruits spending exactly $13, how many apples did he buy?"

# Define 10 different chain-of-thought reasoning paths
reasoning_paths = [
    {
        "name": "Algebraic (Correct)",
        "steps": [
            "Let a = apples, o = oranges",
            "Equation 1: a + o = 5 (total fruits)",
            "Equation 2: 3a + 2o = 13 (total cost)",
            "From eq1: o = 5 - a",
            "Substitute: 3a + 2(5-a) = 13",
            "3a + 10 - 2a = 13",
            "a = 3",
            "Check: 3 apples ($9) + 2 oranges ($4) = $13 ✓"
        ],
        "answer": 3,
        "confidence": 0.95,
        "category": "systematic"
    },
    {
        "name": "Trial and Error (Correct)",
        "steps": [
            "Try 1 apple: $3, need 4 oranges=$8, total=$11 ✗",
            "Try 2 apples: $6, need 3 oranges=$6, total=$12 ✗",
            "Try 3 apples: $9, need 2 oranges=$4, total=$13 ✓",
            "3 apples + 2 oranges = 5 fruits ✓",
            "Answer: 3 apples"
        ],
        "answer": 3,
        "confidence": 0.88,
        "category": "empirical"
    },
    {
        "name": "Pattern Recognition (Correct)",
        "steps": [
            "Notice: Need $13 from 5 fruits",
            "Average price would be $13/5 = $2.60",
            "Oranges=$2, Apples=$3, so need more apples",
            "Price difference: $1 per substitution",
            "All oranges: 5×$2=$10, need $3 more",
            "Replace 3 oranges with 3 apples: adds $3",
            "Answer: 3 apples"
        ],
        "answer": 3,
        "confidence": 0.82,
        "category": "heuristic"
    },
    {
        "name": "Averaging Error",
        "steps": [
            "Average fruit price: ($3+$2)/2 = $2.50",
            "Total spent: $13",
            "Number of fruits: $13/$2.50 = 5.2 ≈ 5 ✓",
            "Since apples are more expensive, roughly half",
            "5/2 = 2.5, round up to 3",
            "Answer: 3 apples"
        ],
        "answer": 3,
        "confidence": 0.45,
        "category": "approximate"
    },
    {
        "name": "Greedy Algorithm (Wrong)",
        "steps": [
            "Buy expensive items first to reach $13",
            "4 apples = $12 (1 fruit left)",
            "1 orange = $2",
            "Total: $14 > $13 ✗",
            "Try 4 apples anyway",
            "Answer: 4 apples"
        ],
        "answer": 4,
        "confidence": 0.35,
        "category": "greedy"
    },
    {
        "name": "Ratio Reasoning (Wrong)",
        "steps": [
            "Price ratio apples:oranges = 3:2",
            "So buy in ratio 3:2",
            "Total 5 fruits: 3 apples, 2 oranges",
            "Cost: 3×$3 + 2×$2 = $13 ✓",
            "Answer: 3 apples"
        ],
        "answer": 3,
        "confidence": 0.72,
        "category": "heuristic"
    },
    {
        "name": "Backward Chaining (Correct)",
        "steps": [
            "Target: $13 with 5 fruits",
            "Maximum spend (all apples): 5×$3=$15",
            "Minimum spend (all oranges): 5×$2=$10",
            "Need exactly $13 (middle ground)",
            "Gap from minimum: $13-$10=$3",
            "Each apple adds $1 vs orange",
            "Need 3 apples, 2 oranges",
            "Answer: 3 apples"
        ],
        "answer": 3,
        "confidence": 0.90,
        "category": "systematic"
    },
    {
        "name": "Assumption Error",
        "steps": [
            "Assume equal distribution",
            "5 fruits / 2 types ≈ 2.5 each",
            "Round up: 3 apples, 2 oranges",
            "Check: doesn't verify total cost",
            "Answer: 3 apples"
        ],
        "answer": 3,
        "confidence": 0.40,
        "category": "approximate"
    },
    {
        "name": "Calculation Error",
        "steps": [
            "Let a=apples, o=oranges",
            "a + o = 5",
            "3a + 2o = 13",
            "From first: o = 5 - a",
            "Substitute: 3a + 2(5-a) = 13",
            "3a + 10 - 2a = 13",
            "a = 13 - 10 = 3... wait, a = 2?",
            "Answer: 2 apples"
        ],
        "answer": 2,
        "confidence": 0.50,
        "category": "systematic_error"
    },
    {
        "name": "Visual Grouping (Correct)",
        "steps": [
            "Draw 5 circles for fruits",
            "Need $13 total",
            "Try groups: [A,A,A,O,O]",
            "Cost: $3+$3+$3+$2+$2=$13 ✓",
            "Count apples: 3",
            "Answer: 3 apples"
        ],
        "answer": 3,
        "confidence": 0.85,
        "category": "visual"
    }
]

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
    print("Generating Chain-of-Thought Visualizations...")
    print(f"Analyzing {len(reasoning_paths)} different reasoning strategies\n")

    # Print summary
    print_summary(reasoning_paths)

    # Create embeddings
    print("Creating latent space embeddings...")
    embeddings_2d = create_embeddings(reasoning_paths)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_reasoning_paths(reasoning_paths, 'reasoning_paths.png')
    plot_latent_space(reasoning_paths, embeddings_2d, 'latent_space.png')
    plot_answer_distribution(reasoning_paths, 'answer_distribution.png')

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. reasoning_paths.png - Detailed flowcharts of all reasoning paths")
    print("  2. latent_space.png - 2D projection of reasoning embeddings")
    print("  3. answer_distribution.png - Answer frequency and confidence analysis")
    print("\n" + "="*80)
