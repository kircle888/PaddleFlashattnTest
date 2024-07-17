import paddle
import paddle.nn.functional as F
import sys

sys.path.append("../")
from paddle_funcs import genqkv, time_it, unpad_input, pad_input
import argparse
import json
from pathlib import Path
import numpy as np
import math


def get_inputs(sq, sk, hq, hk, hdim, dtype):
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
        key_padding_mask = paddle.zeros((bsz, sq), dtype="int32")
        for i in range(bsz):
            key_padding_mask[i, : seqlens[i]] = 1
        yield q, k, v, cu_seqlens, key_padding_mask


def unpad_it(x, key_padding_mask):
    head_dim = x.shape[-1]
    x = x.flatten(2, 3)
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
    x_unpad = x_unpad.reshape([x_unpad.shape[0], -1, head_dim])
    return x_unpad, indices, cu_q_lens, max_s


def time_flashattn_unpad(sq, sk, hq, hk, hdim, causal, dtype):
    assert sq == sk
    max_s = sq

    times = []
    gener = get_inputs(sq, sk, hq, hk, hdim, dtype)
    for q, k, v, cu_seqlens, key_padding_mask in gener:
        bsz = q.shape[0]

        def fa_func():
            q_unpad, indices, _, _ = unpad_it(q, key_padding_mask)
            k_unpad, _, _, _ = unpad_it(k, key_padding_mask)
            v_unpad, _, _, _ = unpad_it(v, key_padding_mask)
            o_unpad, _ = F.flash_attention.flash_attn_unpadded(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens,
                cu_seqlens,
                max_s,
                max_s,
                scale=1 / math.sqrt(hdim),
                dropout=0.0,
                causal=causal,
            )
            o_unpad = paddle.flatten(o_unpad, 1, 2)
            o = pad_input(o_unpad, indices, bsz, sq)
            o = o.reshape([bsz, sq, hq, hdim])
            o[0].backward(retain_graph=True)
            return o[0]

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
    t = time_flashattn_unpad(*case)
    with open(args.log_file, "a") as f:
        print("unpad", *case, file=f, flush=True)
        print(t, file=f, flush=True)
