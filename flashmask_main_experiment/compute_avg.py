import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df2 = pd.read_csv("logs5/table.csv")
",run_mode,bsz,sq,sk,hq,hk,hdim,sparsity,qaratio,causal,has_end,time(ms)"


def mul_samples_ratio(row):
    qaratio = row["qaratio"]
    seqlen = row["sq"]
    dataname = f"data/srowsQ{qaratio}A{seqlen//1024}K.npz"
    d = np.load(dataname)
    sample_num = sum([d[k].shape[0] for k in d])
    if str(row['sparsity']) not in d:
        import pdb
        pdb.set_trace()
    row_num = d[str(row["sparsity"])].shape[0]
    return row["time(ms)"] * row_num / sample_num


df2["time(ms)"] = df2.apply(mul_samples_ratio, axis=1)


df2 = (
    df2.groupby(
        [
            "run_mode",
            "bsz",
            "sq",
            "sk",
            "hq",
            "hk",
            "hdim",
            "qaratio",
            "causal",
            "has_end",
        ]
    )["time(ms)"]
    .sum()
    .reset_index()
)


print(df2)

df = pd.read_csv("kernel_time.csv")
df = df[df["kernel"].isin(["attentionmask", "math_attention"])]
df = df[df["bsz"] == 1]
del df["seed"]
df: pd.DataFrame = df.assign(qaratio=df["dataname"].apply(lambda x: int(x[6])))
df = df.assign(run_mode=df["kernel"])
del df["kernel"]
del df["dataname"]
del df["hm"]
del df["sparsity"]
df = df.assign(causal=True)
df = df.assign(has_end=False)
del df["Unnamed: 0"]
df.reset_index()
df.to_csv("dense.csv")

df = pd.concat([df, df2], ignore_index=True)
print(df)
df.to_csv("logs5/avg.csv")
df.to_markdown("logs5/avg.md")
