import paddle
import paddle.nn.functional as F
import sys

sys.path.append("../")
from paddle_funcs import genqkv, time_it, packqkv, unpackqkv, clone_tensors
import argparse


def time_flashattn(bsz, sq, sk, hq, hk, hdim, causal, dtype):
    qkv = packqkv(*genqkv(bsz, sq, sk, hq, hk, hdim, dtype))
    (qkv,) = clone_tensors([qkv])

    def fa_func():
        q, k, v = unpackqkv(qkv)
        o = F.flash_attention.flash_attention(q, k, v, causal=causal)
        o[0].backward()
        return o[0]

    t = time_it(400, 20, fa_func)
    return t


def time_qkvpacked(bsz, sq, sk, hq, hk, hdim, causal, dtype):
    qkv = packqkv(*genqkv(bsz, sq, sk, hq, hk, hdim, dtype))
    (qkv,) = clone_tensors([qkv])

    def fa_func():
        o = F.flash_attention.flash_attn_qkvpacked(qkv, causal=causal)
        o[0].backward()
        return o[0]

    t = time_it(400, 20, fa_func)
    return t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_mode", type=str, choices=["qkvpacked", "notpacked"])
    parser.add_argument("case_id", type=int)
    parser.add_argument("device", type=str)
    parser.add_argument("log_file", type=str)
    args = parser.parse_args()
    paddle.set_device(args.device)
    from cases import cases

    case = cases[args.case_id]
    if args.run_mode == "qkvpacked":
        t = time_qkvpacked(*case)
    else:
        t = time_flashattn(*case)
    with open(args.log_file, "a") as f:
        print(args.run_mode, *case, file=f, flush=True)
        print(t, file=f, flush=True)
