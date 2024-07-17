import paddle
import paddle.nn.functional as F
import sys

sys.path.append("../")
from paddle_funcs import genqkv, time_it, varlen_to_startendrows
import argparse
import json
from pathlib import Path
import numpy as np
import math


def get_inputs(sq, sk, hq, hk, hdim, causal, dtype):
    if sq < 1024:
        ks = sq / 1024
        ks = f"{ks :2}"
    else:
        ks = sq // 1024
        ks = str(ks)
    with Path(".", "data", f"metadataQ1A{ks}K.json").open("r") as f:
        d = json.load(f)
    for item in d:
        seqlens = [sum(x) for x in item]
        bsz = len(seqlens)
        q, k, v = genqkv(bsz, sq, sk, hq, hk, hdim, dtype)
        seqlens = np.array(seqlens, dtype=np.int32)
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), np.cumsum(seqlens)])
        cu_seqlens = paddle.to_tensor(cu_seqlens, dtype="int32")
        startend_rows = varlen_to_startendrows(cu_seqlens, bsz, sq, causal)
        yield q, k, v, startend_rows


def time_flashattn_unpad(sq, sk, hq, hk, hdim, causal, dtype):
    assert sq == sk
    max_s = sq

    times = []
    gener = get_inputs(sq, sk, hq, hk, hdim, causal, dtype)
    for q, k, v, startend_rows in gener:

        def fa_func():
            o = F.flash_attention.flashmask_attention(
                q,
                k,
                v,
                startend_rows,
                causal=causal,
            )
            o.backward(retain_graph=True)
            return o

        t = time_it(50, 5, fa_func)
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
    t = time_flashattn_unpad(*case)
    with open(args.log_file, "a") as f:
        print("varlen", *case, file=f, flush=True)
        print(t, file=f, flush=True)
