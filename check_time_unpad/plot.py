import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path


def parse_log(log_file):
    with open(log_file, "r") as f:
        lines = f.readlines()
    title = None
    v = []
    for line in lines:
        if title is None:
            title = line.strip()
        else:
            item = title.split(" ")
            item.append(float(line) * 1000)
            v.append(item)
            title = None
    return v


logs = Path("logs").resolve().glob("rank*.log")
table = []
for log in logs:
    table.extend(parse_log(log))
df = pd.DataFrame(
    table,
    columns=[
        "run_mode",
        "bsz",
        "sq",
        "sk",
        "hq",
        "hk",
        "hdim",
        "causal",
        "dtype",
        "time(ms)",
    ],
)
sns.set_theme()
sns.set(font_scale=1.5)
df["xlabel"] = df["bsz"] + "-" + df["hq"] + "-" + df["hdim"]
valid_labels = ["1-8-32", "1-8-128", "4-8-128", "1-32-32"]
df = df[df["xlabel"].isin(valid_labels)]
df["sq"] = df["sq"].astype(int)
df = df.sort_values(by=["causal", "sq"], ascending=[True, True])

groups = df.groupby(["causal", "sq"])
ncols = 6
nrows = (len(groups) + 5) // 6
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 8 * nrows))
axes = axes.flatten()
for (causal, sq), (name, group), ax in zip(groups.groups.keys(), groups, axes):
    hue_order = ["notpacked", "qkvpacked"]
    colors = ["#5b9bd5", "#C76DA2"]
    sns.barplot(
        x="xlabel", y="time(ms)", hue="run_mode", data=group, ax=ax, palette=colors
    )
    ax.set_title(f"seqlen={sq},causal={causal}")
    ax.set_xlabel("case")
    ax.set_ylabel("time(ms)")
plt.tight_layout()
plt.savefig("logs/fig.png")

grouped_mean = df.groupby(["causal", "sq", "run_mode"])["time(ms)"].mean().reset_index()
grouped_mean = grouped_mean.pivot_table(
    index=["sq", "causal"],
    columns="run_mode",
    values="time(ms)",
    aggfunc="first",
).reset_index()
grouped_mean: pd.DataFrame = grouped_mean.assign(
    acc=grouped_mean["notpacked"] / grouped_mean["qkvpacked"] - 1
)
grouped_mean["acc"] = grouped_mean["acc"].apply(lambda x: f"{x:.2%}")
grouped_mean.to_markdown("logs/mean.md")
grouped_mean.to_csv("logs/mean.csv")
