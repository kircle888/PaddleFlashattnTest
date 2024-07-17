import paddle
import paddle.nn.functional as F
import numpy as np
import time
import sys

sys.path.append("../")
from paddle_funcs import (
    flashmask_to_densemask,
    clone_tensors,
    check,
    strict_check,
    get_fa_argparser,
    parse_h,
    print_config,
    genqkv,
    genograd,
    gen_qa_mask,
    gen_random_flashmask,
)


def get_fafunc(
    run_mode,
    q,
    k,
    v,
    startend_row_indices,
    dropout,
    causal,
    ograd,
):
    if run_mode == "densemask":
        attn_mask = flashmask_to_densemask(
            startend_row_indices,
            q.dtype,
            causal=causal,
        )

        def target_func():
            q1, k1, v1 = clone_tensors([q, k, v])
            out = F.scaled_dot_product_attention(
                q1,
                k1,
                v1,
                attn_mask=attn_mask,
                is_causal=False,
            )
            out.backward(ograd)
            return out, q1.grad, k1.grad, v1.grad

    elif run_mode == "flashmask":

        def target_func():
            q1, k1, v1 = clone_tensors([q, k, v])
            out = F.flashmask_attention(
                q1,
                k1,
                v1,
                startend_row_indices=startend_row_indices,
                dropout=dropout,
                causal=causal,
            )
            paddle.device.synchronize()
            out.backward(ograd)
            paddle.device.synchronize()
            return out, q1.grad, k1.grad, v1.grad

    else:
        assert 0
    return target_func


def run_densemask(
    bsz,
    sq,
    sk,
    hq,
    hk,
    hm,
    hdim,
    causal=True,
    has_end=False,
    mask_gen="random",
    log_file="sparsefa.log",
):
    with open(log_file, "a") as f:
        print(
            f"{bsz} {sq} {sk} {hq} {hk} {hm} {hdim} {causal} {has_end} {mask_gen}",
            flush=True,
            file=f,
        )
    dtype = "float16"
    dropout = 0.0
    # test dynamic
    paddle.disable_static()
    paddle.set_flags({"FLAGS_cudnn_deterministic": 1})

    q, k, v = genqkv(bsz, sq, sk, hq, hk, hdim, dtype)
    ograd = genograd(bsz, sq, hq, hdim, dtype)

    if mask_gen == "random":
        startend_row_indices = gen_random_flashmask(bsz, hm, sq, has_end, causal)
    elif mask_gen == "qamask":
        startend_row_indices = gen_qa_mask(bsz, hm, sq, has_end, causal)
    else:
        raise ValueError(f"{mask_gen} is not a valid value")
    fn = get_fafunc(
        "densemask",
        q,
        k,
        v,
        startend_row_indices=startend_row_indices,
        dropout=dropout,
        causal=causal,
        ograd=ograd,
    )
    out, qgrad, kgrad, vgrad = fn()

    np.savez(
        log_file.replace(".log", ".npz"),
        q=q.cast("float32").numpy(),
        k=k.cast("float32").numpy(),
        v=v.cast("float32").numpy(),
        ograd=ograd.cast("float32").numpy(),
        startend_row_indices=startend_row_indices,
        out=out.cast("float32").numpy(),
        qgrad=qgrad.cast("float32").numpy(),
        kgrad=kgrad.cast("float32").numpy(),
        vgrad=vgrad.cast("float32").numpy(),
    )
    with open(log_file, "a") as f:
        print(
            "Std result for shape saved",
            flush=True,
            file=f,
        )


def run_flashmask(
    bsz,
    sq,
    sk,
    hq,
    hk,
    hm,
    hdim,
    causal=True,
    has_end=False,
    mask_gen="random",
    log_file="sparsefa.log",
):
    dtype = "float16"
    dropout = 0.0
    # test dynamic
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
    startend_row_indices = paddle.to_tensor(d["startend_row_indices"], dtype="int32")

    fn = get_fafunc(
        run_mode,
        q,
        k,
        v,
        startend_row_indices=startend_row_indices,
        dropout=dropout,
        causal=causal,
        ograd=ograd,
    )
    out, qgrad, kgrad, vgrad = fn()

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
    parser = get_fa_argparser()
    parser.add_argument(
        "run_mode",
        choices=[
            "densemask",
            "flashmask",
        ],
        help="run mode",
    )
    parser.add_argument("case_id", type=int)
    parser.add_argument("device", type=str)
    parser.add_argument("log_file", type=str)
    args = parser.parse_args()
    run_mode = args.run_mode
    paddle.set_device(args.device)
    from cases import cases

    case = cases[args.case_id]
    if run_mode == "densemask":
        run_densemask(*case, log_file=args.log_file)
        hash_file = args.log_file.replace(".log", ".hash")
        data_file = args.log_file.replace(".log", ".npz")
        os.system(f"md5sum {data_file} >>{hash_file}")

    elif run_mode == "flashmask":
        run_flashmask(*case, log_file=args.log_file)
