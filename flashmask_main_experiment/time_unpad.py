import paddle
import paddle.nn.functional as F
import sys

sys.path.append("../")
from paddle_funcs import genqkv, time_it
import argparse
import json
from pathlib import Path
import numpy as np
import math


def get_inputs(sq, sk, hq, hk, hdim, dtype, sparsity):
    if sq < 1024:
        ks = sq / 1024
        ks = f"{ks :2}"
    else:
        ks = sq // 1024
        ks = str(ks)
    if not Path("data",f"srowsQ1A{ks}K.npz").exists():
        exit(0)
    d = np.load(f"data/srowsQ1A{ks}K.npz")
    if str(sparsity) not in d:
        exit(0)
    d = d[str(sparsity)]
    for item in d:
        cu_seqlens = np.unique(item)
        q, k, v = genqkv(1, sq, sk, hq, hk, hdim, dtype)
        q = q.flatten(0, 1)
        k = k.flatten(0, 1)
        v = v.flatten(0, 1)
        cu_seqlens = paddle.to_tensor(cu_seqlens, dtype="int32")
        yield q, k, v, cu_seqlens


def time_flashattn_unpad(
    run_mode,
    bsz,
    sq,
    sk,
    hq,
    hk,
    hdim,
    sparsity,
    qaratio,
    causal=True,
    has_end=False,
):
    assert sq == sk
    assert qaratio == 1
    assert bsz == 1
    max_s = sq
    dtype = "bfloat16"

    times = []
    gener = get_inputs(sq, sk, hq, hk, hdim, dtype, sparsity)
    for q, k, v, cu_seqlens in gener:

        def fa_func():
            o, _ = F.flash_attention.flash_attn_unpadded(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_s,
                max_s,
                scale=1 / math.sqrt(hdim),
                dropout=0.0,
                causal=causal,
            )
            o.backward(retain_graph=True)
            return o

        t = time_it(50, 10, fa_func)
        times.append(t)
    return np.average(times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id", type=int)
    parser.add_argument("device", type=str)
    parser.add_argument("log_file", type=str)
    args = parser.parse_args()
    paddle.set_device(args.device)
    from cases import cases

    case = cases[args.case_id]
    # print(case)
    t = time_flashattn_unpad("unpad", *case)
    with open(args.log_file, "a") as f:
        print("unpad", *case, file=f, flush=True)
        print(t, file=f, flush=True)
