import matplotlib.pyplot as plt

# Model names and accuracies
models = [
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen2.5-1.5B-Instruct",
    "DeepScaleR-1.5B-Preview",
    "Nemotron-1.5B",
    "ZR1-1.5B"
]

accuracies = [91.48, 50.93, 91.57, 87.77, 86.23]

# Define moderate color palette
colors = ["#6baed6", "#9ecae1", "#74c476", "#fd8d3c", "#969696"]

plt.figure(figsize=(18, 10))
bars = plt.bar(models, accuracies, color=colors, edgecolor="black")

# Add accuracy labels above each bar
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{acc:.2f}%", ha='center', va='bottom', fontsize=20)

plt.title("Accuracy (Valid) of 1.5B Math Models (Pass@1 on MATH-500)", fontsize=28, weight="bold")
plt.ylabel("Accuracy (%)", fontsize=24)
plt.xticks(rotation=20, fontsize=20)
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("1.5bmodel_acc_math.png")