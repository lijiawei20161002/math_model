import matplotlib.pyplot as plt

# Data
tokens = [1024, 2048, 3072, 4096, 5120, 10240]

acc_1 = [3.33, 6.67, 20.00, 20.00, 30.00, 23.33]
acc_5 = [3.33, 0.00, 20.00, 30.00, 26.67, 30.00]
acc_10 = [3.33, 10.00, 26.67, 26.67, 30.00, 33.33]
acc_20 = [0.00, 10.00, 26.67, 26.67, 30.00, 33.33]

# Plot
plt.figure(figsize=(12, 7))

plt.plot(tokens, acc_1, marker="o", linewidth=3, label="samples=1, temp=0")
plt.plot(tokens, acc_5, marker="s", linewidth=3, label="samples=5, temp=0.7")
plt.plot(tokens, acc_10, marker="^", linewidth=3, label="samples=10, temp=0.7")
plt.plot(tokens, acc_20, marker="D", linewidth=3, label="samples=20, temp=0.7")

# Beautify
plt.title("Accuracy (Pass@1 Correctness) vs Max Tokens", fontsize=18, weight="bold")
plt.xlabel("Max Tokens", fontsize=14)
plt.ylabel("Accuracy (Pass@1%)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(tokens, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc="best")
plt.tight_layout()
plt.savefig("deepscaler_majority_vote_vs_single.png")