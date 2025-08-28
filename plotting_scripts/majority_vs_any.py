import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Data
tokens = [10240, 20480, 30720]

data = {
    10: {
        "ANY": [46.67, 53.33, 50.00],
        "MV":  [33.33, 30.00, 40.00],
    },
    20: {
        "ANY": [50.00, 56.67, 50.00],
        "MV":  [33.33, 36.67, 33.33],
    },
    30: {
        "ANY": [50.00, 50.00, 56.67],
        "MV":  [33.33, 43.33, 36.67],
        "runtime_note": "52:59"  # Only this runtime label requested
    }
}

# Colors for samples_per_question
color_map = {10: "#1f77b4", 20: "#2ca02c", 30: "#d62728"}  # blue, green, red
linestyle_map = {"ANY": "-", "MV": "--"}
marker_map = {"ANY": "o", "MV": "s"}

plt.figure(figsize=(20, 14))

# Plot lines
for spq, results in data.items():
    for key in ["ANY", "MV"]:
        y = results[key]
        plt.plot(
            tokens, y,
            linestyle=linestyle_map[key],
            marker=marker_map[key],
            color=color_map[spq],
            linewidth=5,
            markersize=12,
        )
        # Text labels along lines
        for x_val, y_val in zip(tokens, y):
            dy = 2.5 if key == "ANY" else -3.0
            plt.annotate(f"{y_val:.2f}%",
                         xy=(x_val, y_val),
                         xytext=(0, dy*3),
                         textcoords="offset points",
                         ha="center",
                         fontsize=22,
                         fontweight="bold")

# Runtime annotation (only one label requested: 52:59 at max token=30720 for spq=30)
runtime_x = 30720
runtime_text = data[30].get("runtime_note", None)
if runtime_text:
    plt.annotate(f"runtime {runtime_text}",
                 xy=(runtime_x, 60),
                 xytext=(20, 20),
                 textcoords="offset points",
                 fontsize=22,
                 fontweight="bold",
                 color="black")

# Custom legend
legend_elements = [
    Line2D([0], [0], color=color_map[10], lw=5, label="samples=10"),
    Line2D([0], [0], color=color_map[20], lw=5, label="samples=20"),
    Line2D([0], [0], color=color_map[30], lw=5, label="samples=30"),
    Line2D([0], [0], color="black", lw=5, linestyle="-", label="ANY (solid)"),
    Line2D([0], [0], color="black", lw=5, linestyle="--", label="Majority Vote (dashed)"),
]

plt.legend(handles=legend_elements, fontsize=22, ncol=2)

# Formatting
plt.title(
    "Majority Vote vs ANY — agentica-org/DeepScaleR-1.5B-Preview on AIME 2025\n(pass@1 correctness, T=0.6, top-p=0.95)",
    fontsize=28, fontweight="bold"
)
plt.xlabel("Max tokens", fontsize=26, fontweight="bold")
plt.ylabel("Acc(valid) — pass@1 (%)", fontsize=26, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.4)
plt.xticks(tokens, fontsize=22)
plt.yticks(fontsize=22)
plt.tight_layout()
plt.savefig("majority_vote_vs_any.png")