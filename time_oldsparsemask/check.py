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


"flashmask 32768 32768 32 32 128 True bfloat16 2"
if __name__ == "__main__":
    logs = Path("logs").resolve().glob("rank*.log")
    table = []
    for log in logs:
        table.extend(parse_log(log))
    df = pd.DataFrame(
        table,
        columns=[
            "run_mode",
            "sq",
            "sk",
            "hq",
            "hk",
            "hdim",
            "causal",
            "dtype",
            "qaratio",
            "time(ms)",
        ],
    )

    def get_mode_name(x):
        x = int(x)
        if x == 1:
            return "sft"
        elif x == 2:
            return "dpo"
        else:
            return "rm"

    df["mode"] = df["qaratio"].apply(get_mode_name)
    del df["qaratio"]
    df.to_csv("logs/table.csv")
    df = df.pivot_table(
        index=["sq", "sk", "hq", "hk", "hdim", "causal", "dtype", "mode"],
        columns="run_mode",
        values="time(ms)",
        aggfunc="first",
    ).reset_index()
    df: pd.DataFrame = df.assign(acc=df["oldsparsemask"] / df["flashmask"] - 1)
    df["acc"] = df["acc"].apply(lambda x: f"{x:.2%}")
    df.to_markdown("logs/table.md")
    # df.to_csv("logs/table.csv")
    print(df)
