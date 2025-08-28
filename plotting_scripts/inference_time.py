import matplotlib.pyplot as plt

# Tokens
tokens = [10240, 20480, 30720]

# Inference runtimes from your log (MM:SS format)
# Each samples_per_question has one runtime per token
runtime_data = {
    10: ["05:14", "27:42", "36:18"],
    20: ["09:12", "35:06", "44:11"],
    30: ["25:11", "43:58", "52:59"]
}

# Convert times (MM:SS) to total seconds (for plotting)
def to_seconds(tstr):
    m, s = map(int, tstr.split(":"))
    return m*60 + s

runtime_seconds = {
    spq: [to_seconds(t) for t in times]
    for spq, times in runtime_data.items()
}

# Colors
color_map = {10: "#1f77b4", 20: "#2ca02c", 30: "#d62728"}

plt.figure(figsize=(20, 14))

# Plot
for spq, times_sec in runtime_seconds.items():
    plt.plot(tokens, times_sec,
             marker="o",
             linewidth=5,
             markersize=12,
             color=color_map[spq],
             label=f"samples={spq}")
    # Add runtime text labels
    for x, sec, txt in zip(tokens, times_sec, runtime_data[spq]):
        plt.annotate(txt,
                     xy=(x, sec),
                     xytext=(0, 15),
                     textcoords="offset points",
                     ha="center",
                     fontsize=22,
                     fontweight="bold")

# Formatting
plt.title("Total Inference Time — agentica-org/DeepScaleR-1.5B-Preview on AIME 2025\n 7 H100 with vLLM",
          fontsize=28, fontweight="bold")
plt.xlabel("Max tokens", fontsize=26, fontweight="bold")
plt.ylabel("Total Inference Time (seconds)", fontsize=26, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.4)
plt.xticks(tokens, fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=22)
plt.tight_layout()
plt.savefig("deepscaler_inference_time.png")