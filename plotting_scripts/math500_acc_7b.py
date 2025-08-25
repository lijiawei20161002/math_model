import matplotlib.pyplot as plt

# Data
models = [
    "DeepSeek-R1-Distill-Qwen-7B",
    "Qwen2.5-7B-Instruct",
    "MetaMath-Mistral-7B",
    "Mistral-7B-Instruct-v0.3",
    "Llama-2-7B-hf"
]
accuracy = [90.62, 72.28, 29.73, 22.22, 0.00]

# Plot
plt.figure(figsize=(10,6))
bars = plt.bar(models, accuracy, width=0.5, color=plt.cm.Set2.colors)  # moderate colors

# Add labels on top of bars
for bar, acc in zip(bars, accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{acc:.2f}%", ha='center', va='bottom', fontsize=14)

# Styling
plt.title("Validation Accuracy (Pass@1 %) of Different Models", fontsize=14)
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.xticks(rotation=20, ha='right')

plt.tight_layout()
plt.savefig("7bmodel_acc_math500.png")