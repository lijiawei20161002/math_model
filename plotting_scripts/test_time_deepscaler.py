# -*- coding: utf-8 -*-
"""
Publication-style AIME 2025 plot
Accuracy vs Pass@k grouped by Max Token length
"""

import matplotlib.pyplot as plt
from collections import defaultdict

# Raw data (k, max_token, any_acc%, mv_acc%)
raw = [
    (10, 10240, 46.67, 33.33), (10, 20480, 53.33, 30.00), (10, 30720, 50.00, 40.00),
    (20, 10240, 50.00, 33.33), (20, 20480, 56.67, 36.67), (20, 30720, 50.00, 33.33),
    (30, 10240, 50.00, 33.33), (30, 20480, 50.00, 43.33), (30, 30720, 56.67, 36.67),
    (100, 30720, 60.00, 36.67), (100, 51200, 60.00, 36.67),
    (200, 20480, 63.33, 40.00),
    (500, 4096, 43.33, 30.00), (500, 10240, 70.00, 36.67),
    (1000, 10240, 76.67, 36.67), (1000, 20480, 70.00, 36.67),
]

# Organize by token length
by_tok_any, by_tok_mv = defaultdict(list), defaultdict(list)
for k, tok, any_p, mv_p in raw:
    by_tok_any[tok].append((k, any_p / 100))
    by_tok_mv[tok].append((k, mv_p / 100))

for d in (by_tok_any, by_tok_mv):
    for tok in d:
        d[tok].sort(key=lambda x: x[0])

token_lengths = sorted(by_tok_any.keys())

# Colors/markers
colors = plt.cm.tab10.colors
markers = ["o", "s", "D", "^", "v", "P", "X", "h"]

# Style
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.facecolor": "#f4f4f4",
    "figure.facecolor": "white",
    "grid.color": "#bbbbbb",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Helper function
def plot_curves(ax, data_dict, title):
    for i, tok in enumerate(token_lengths):
        k_vals = [k for k, _ in data_dict[tok]]
        y_vals = [y for _, y in data_dict[tok]]
        ax.plot(
            k_vals, y_vals,
            label=f"Max token {tok:,}",
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            markersize=7, linewidth=2,
        )
    ax.set_xscale("log")
    all_k = sorted(set(k for tok in data_dict for k, _ in data_dict[tok]))
    ax.set_xticks(all_k)
    ax.set_xticklabels([str(k) for k in all_k], rotation=35)
    ax.set_xlim(left=10)  # <-- Start at 10 samples, remove left blank
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.grid(True)
    ax.set_title(title)
    ax.legend(ncol=3, frameon=True)

plot_curves(ax1, by_tok_any, "Pass@k (ANY) vs Samples per Question")
plot_curves(ax2, by_tok_mv, "Majority Vote Accuracy vs Samples per Question")

ax2.set_xlabel("Samples per Question (Pass@k)")
plt.tight_layout()
plt.savefig("aime_tokens_passatk.png", dpi=300)