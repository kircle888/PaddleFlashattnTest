import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("logs5/table.csv")
df = df[df["sq"] == 32 * 1024]


def get_mode(qa):
    if qa == 1:
        return "sft"
    elif qa == 2:
        return "dpo"
    elif qa == 6:
        return "rm"
    else:
        raise ValueError(f"Invalid QA {qa}")


df["run_mode"] = df["run_mode"].map({"flashmask": "FlashMask", "unpad": "FA-Varlen"})
df["Latency(ms)"] = df["time(ms)"]
df["Sparsity(%)"] = df["sparsity"] * 10
df["mode"] = df["qaratio"].apply(get_mode)
df.drop(columns=["time(ms)", "sparsity", "qaratio"], inplace=True)
sns.set_theme()
sns.set(font_scale=1.5)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
for i, mode in enumerate(["sft", "dpo", "rm"]):
    xdf = df[df["mode"] == mode].copy()
    # print(xdf)
    hue_order = ["FA-Varlen", "FlashMask"]
    colors = ["#5b9bd5", "#C76DA2"]
    markers = ["^", "o"]
    if mode != "sft":
        hue_order = hue_order[1:]
        colors = colors[1:]
        markers = markers[1:]
    sns.lineplot(
        data=xdf,
        hue="run_mode",
        style="run_mode",
        x="Sparsity(%)",
        y="Latency(ms)",
        ax=axes[i],
        hue_order=hue_order,
        style_order=hue_order,
        palette=colors,
        markers=markers,
        markersize=10,
        dashes=False
    )
    axes[i].set_ylim((0, 220))
    axes[i].set_xlim((-5, 100))
    axes[i].get_legend().set_title("")
    axes[i].set_title(f"{mode.upper()}")
fig.tight_layout()
plt.savefig("logs5/sparsity_kernel_time.png")
# plt.savefig("sparsity_kernel_time.pdf")
