import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path


df = pd.read_csv("logs5/avg.csv")
sns.set_theme()
sns.set(font_scale=1.5)
df["sq"] = df["sq"].astype(int)
df = df.sort_values(by=["causal", "sq"], ascending=[True, True])


def rename(x):
    if x == "math_attention":
        return "VanillaAttention"
    elif x == "attentionmask":
        return "DenseMask"
    elif x == "flashmask":
        return "FlashMask"
    elif x == "unpad":
        return "FA-Varlen"


df["run_mode"] = df["run_mode"].apply(rename)
groups = df.groupby(["qaratio"])
ncols = 3
nrows = 1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 8 * nrows))
axes = axes.flatten()
for (qaratio), (name, group), ax in zip(groups.groups.keys(), groups, axes):
    hue_order = ["FA-Varlen", "VanillaAttention", "DenseMask", "FlashMask"]
    colors = ["#32B897", "#F27970", "#8983BF", "#C76DA2"]
    if qaratio != 1:
        hue_order = hue_order[1:]
        colors = colors[1:]
    sns.barplot(
        x="sq",
        y="time(ms)",
        hue="run_mode",
        data=group,
        ax=ax,
        hue_order=hue_order,
        palette=colors,
    )
    if qaratio == 1:
        mode = "SFT"
    elif qaratio == 2:
        mode = "DPO"
    else:
        mode = "RM"
    ax.get_legend().set_title("")
    ax.set_title(f"{mode}")
    ax.set_xlabel("seqlen")
    ax.set_ylabel("Latency(ms)")
    ax.set_yscale("log")
plt.tight_layout()
plt.savefig("logs5/fig.png")
