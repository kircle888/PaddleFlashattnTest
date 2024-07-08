import paddle
import paddle.nn.functional as F
import sys

sys.path.append("../")
from paddle_funcs import genqkv, genograd, check, strict_check
import argparse
import numpy as np


def save_flashattn(bsz, sq, sk, hq, hk, hdim, causal, dtype, log_file):
    with open(log_file, "a") as f:
        print(
            f"{bsz} {sq} {sk} {hq} {hk} {hdim} {causal} {dtype}",
            flush=True,
            file=f,
        )
    paddle.disable_static()
    paddle.set_flags({"FLAGS_cudnn_deterministic": 1})
    q, k, v = genqkv(bsz, sq, sk, hq, hk, hdim, dtype)
    ograd = genograd(bsz, sq, hq, hdim, dtype)
    o = F.flash_attention.flash_attention(q, k, v, causal=causal)
    out = o[0]
    out.backward(ograd)
    qgrad, kgrad, vgrad = (q.grad, k.grad, v.grad)
    np.savez(
        log_file.replace(".log", ".npz"),
        q=q.cast("float32").numpy(),
        k=k.cast("float32").numpy(),
        v=v.cast("float32").numpy(),
        ograd=ograd.cast("float32").numpy(),
        out=out.cast("float32").numpy(),
        qgrad=qgrad.cast("float32").numpy(),
        kgrad=kgrad.cast("float32").numpy(),
        vgrad=vgrad.cast("float32").numpy(),
    )


def check_flashattn(bsz, sq, sk, hq, hk, hdim, causal, dtype, log_file):
    paddle.disable_static()
    paddle.set_flags({"FLAGS_cudnn_deterministic": 1})
    d = np.load(log_file.replace(".log", ".npz"))
    q = paddle.to_tensor(d["q"], dtype=dtype)
    k = paddle.to_tensor(d["k"], dtype=dtype)
    v = paddle.to_tensor(d["v"], dtype=dtype)
    q.stop_gradient = False
    k.stop_gradient = False
    v.stop_gradient = False
    ograd = paddle.to_tensor(d["ograd"], dtype=dtype)
    ograd.stop_gradient = False
    o = F.flash_attention.flash_attention(q, k, v, causal=causal)
    out = o[0]
    out.backward(ograd)
    qgrad, kgrad, vgrad = (q.grad, k.grad, v.grad)
    try:
        strict_check(out, d["out"])
        strict_check(qgrad, d["qgrad"])
        strict_check(kgrad, d["kgrad"])
        strict_check(vgrad, d["vgrad"])
    except AssertionError as e:
        with open(log_file, "a") as f:
            print("Test Failed", flush=True, file=f)
            print(e, flush=True, file=f)
        return
    with open(log_file, "a") as f:
        print("Test Pass", flush=True, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_mode", type=str)
    parser.add_argument("case_id", type=int)
    parser.add_argument("device", type=str)
    parser.add_argument("log_file", type=str)
    args = parser.parse_args()
    paddle.set_device(args.device)
    from cases import cases

    if args.run_mode == "std":
        save_flashattn(*cases[args.case_id], log_file=args.log_file)
    else:
        check_flashattn(*cases[args.case_id], log_file=args.log_file)
