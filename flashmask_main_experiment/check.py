import tabulate
from pathlib import Path
import numpy as np
import pandas as pd


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


"flashmask 1 2048 2048 32 32 128 9 1 True False"
if __name__ == "__main__":
    logs = Path("logs5").resolve().rglob("rank*.log")
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
            "sparsity",
            "qaratio",
            "causal",
            "has_end",
            "time(ms)",
        ],
    )
    # df = df.pivot_table(
    #     index=["sq", "sk", "hq", "hk", "hdim", "causal", "dtype"],
    #     columns="run_mode",
    #     values="time(ms)",
    #     aggfunc="first",
    # ).reset_index()
    # df: pd.DataFrame = df.assign(acc=df["unpad"] / df["varlen"] - 1)
    # df["acc"] = df["acc"].apply(lambda x: f"{x:.2%}")
    df.to_markdown("logs5/table.md")
    df.to_csv("logs5/table.csv")
    print(df)
