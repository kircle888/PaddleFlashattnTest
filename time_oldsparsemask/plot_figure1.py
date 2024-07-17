import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path


df = pd.read_csv("logs/table.csv")
sns.set_theme()
sns.set(font_scale=1.5)
df["sq"] = df["sq"].astype(int)
df = df[df["hdim"] == 128]
df = df[df["sq"].isin([16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024])]
df = df.sort_values(by=["causal", "sq"], ascending=[True, True])


def rename(x):
    if x == "oldsparsemask":
        return "FlashMask-NoBypass"
    elif x == "flashmask":
        return "FlashMask-Bypass"


df["run_mode"] = df["run_mode"].apply(rename)
group_order = ["sft", "dpo", "rm"]
df["mode"] = pd.Categorical(df["mode"], categories=group_order, ordered=True)
groups = df.groupby(["mode"])
ncols = 3
nrows = 1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 8 * nrows))
axes = axes.flatten()
for (mode), (name, group), ax in zip(groups.groups.keys(), groups, axes):
    print(group)

    hue_order = ["FlashMask-NoBypass", "FlashMask-Bypass"]
    colors = ["#32B897", "#F27970", "#8983BF", "#C76DA2"]
    sns.barplot(
        x="sq",
        y="time(ms)",
        hue="run_mode",
        data=group,
        ax=ax,
        hue_order=hue_order,
        palette=colors,
    )
    ax.set_title(f"{str(mode).upper()}")
    ax.set_xlabel("seqlen")
    ax.set_ylabel("Latency(ms)")
    ax.get_legend().set_title("")
    ax.set_yscale("log")

plt.tight_layout()
plt.savefig("logs/fig.png")
