import matplotlib.pyplot as plt

# Data
samples = [5, 10, 20, 30, 40, 50]
accuracy = [26.67, 26.67, 23.33, 16.67, 23.33, 20.00]

# Plot
plt.figure(figsize=(20,12))  # doubled size
plt.plot(
    samples, accuracy,
    marker="*", linestyle="-", linewidth=5, color='green', markersize=20,
    label="Majority Vote Accuracy"
)

# Add text labels next to each point
for x, y in zip(samples, accuracy):
    plt.text(
        x, y + 0.2, f"{y:.2f}%",
        ha="center", va="bottom", fontsize=24
    )

# Labels and title
plt.xlabel("Samples per Question", fontsize=24)
plt.ylabel("Accuracy (Pass@1%)", fontsize=24)
plt.title("Majority Vote Accuracy — agentica-org/DeepScaleR-1.5B-Preview\n(max_tokens=4096, temperature=0.7)", fontsize=28)
plt.xticks(samples, fontsize=24)
plt.yticks(fontsize=24)
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(fontsize=22)

plt.tight_layout()
plt.savefig("deepscaler_majority_vote.png")