import paddle
import paddle.nn.functional as F
import sys

sys.path.append("../")
from paddle_funcs import genqkv, time_it
import argparse


def time_flashattn(bsz, sq, sk, hq, hk, hdim, causal, dtype):
    q, k, v = genqkv(bsz, sq, sk, hq, hk, hdim, dtype)

    def fa_func():
        o = F.flash_attention.flash_attention(q, k, v, causal=causal)
        o[0].backward()
        return o[0]

    t = time_it(400, 20, fa_func)
    return t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str)
    parser.add_argument("log_file", type=str)
    args = parser.parse_args()
    paddle.set_device(args.device)
    from cases import cases

    for case in cases:
        t = time_flashattn(*case)
        with open(args.log_file, "a") as f:
            print(*case, file=f, flush=True)
            print(t, file=f, flush=True)
