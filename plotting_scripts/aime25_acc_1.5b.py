import matplotlib.pyplot as plt

# Data
models = [
    "Zyphra/ZR1-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "agentica-org/DeepScaleR-1.5B-Preview",
    "nvidia/OpenReasoning-Nemotron-1.5B"
]

# (Correct, Valid)
results = [
    (5, 20),   # Zyphra
    (5, 8),    # DeepSeek
    (0, 28),   # Qwen
    (5, 7),    # DeepScaleR
    (5, 7)     # Nemotron
]

accuracies = [(c / v * 100 if v > 0 else 0) for c, v in results]

# Plot
plt.figure(figsize=(20, 12))
bars = plt.bar(models, accuracies, color=plt.cm.Set2.colors)

# Add text labels (Correct/Valid and %)
for bar, (c, v), acc in zip(bars, results, accuracies):
    label = f"{acc:.2f}%\n{c} correct out of {v} valid"
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             label, ha='center', va='bottom', fontsize=18, weight='bold')

plt.title("AIME 2025 Acc(valid) (pass@1, total 30 questions, max_token=8192)", fontsize=28, weight='bold')
plt.ylabel("Accuracy (%)", fontsize=24)
plt.xticks(rotation=25, ha='right', fontsize=20)

plt.ylim(0, max(accuracies) + 15)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("1.5bmodel_acc_aime25.png")