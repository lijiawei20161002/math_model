import matplotlib.pyplot as plt

# Data
tokens = [1024, 10240, 20480, 30720, 40480]
valid = [0, 8, 9, 20, 21]
correct = [0, 6, 6, 9, 10]
total = [30, 30, 30, 30, 30]
speed = [3.34, 37.30, 87.73, 679.09, 1250.94]  # seconds/it
accuracy = [c/t*100 for c, t in zip(correct, total)]  # % of correct/total

# Create figure
fig, ax1 = plt.subplots(figsize=(20,12))

# Bar plot for valid answers
bars = ax1.bar(tokens, valid, color='gray', width=2500, label="Valid Answers")  
ax1.set_xlabel("Max Tokens", fontsize=24)
ax1.set_ylabel("Valid Answers", color="blue", fontsize=24)
ax1.tick_params(axis="y", labelcolor="blue", labelsize=20)
ax1.tick_params(axis="x", labelsize=20)
ax1.set_xticks(tokens)  # ensure exact values are shown

# Line for accuracy (correct / total)
ax2 = ax1.twinx()
ax2.plot(tokens, accuracy, color="green", marker="s", label="Accuracy (Correct/Total %)")
ax2.set_ylabel("Accuracy (%)", color="green", fontsize=24)
ax2.tick_params(axis="y", labelcolor="green", labelsize=20)

# Add inference speed + accuracy labels
for x, y, s, acc in zip(tokens, accuracy, speed, accuracy):
    ax2.text(x, y+1, f"{acc:.1f}%", ha="center", va="bottom", fontsize=20, color="green")
    ax2.text(x, y-2, f"{s:.0f}s/it", ha="center", va="bottom", fontsize=20, color="blue")

# Legends
ax1.legend(loc="upper left", fontsize=24)
ax2.legend(loc="center left", fontsize=24)

plt.title("agentica-org/DeepScaleR-1.5B-Preview: Valid Answers, Accuracy & Inference Speed", fontsize=24)
plt.tight_layout()
plt.savefig("deepscaler_acc_token_len.png")