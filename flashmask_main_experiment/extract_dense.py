import pandas as pd

df = pd.read_csv("kernel_time.csv")
df = df[df["kernel"].isin(["attentionmask", "math_attention"])]
df = df[df["bsz"] == 1]
del df["seed"]
df: pd.DataFrame = df.assign(qaratio=df["dataname"].apply(lambda x: int(x[6])))
df = df.assign(run_mode=df["kernel"])
del df["kernel"]
del df["dataname"]
del df["hm"]
df = df.assign(causal=True)
df = df.assign(has_end=True)
del df["Unnamed: 0"]
df2 = pd.read_csv("logs/table.csv")
del df2["Unnamed: 0"]
df = pd.concat([df, df2], ignore_index=True)
df.to_csv("logs/table_merged.csv")
df.to_markdown("logs/table_merged.md")
