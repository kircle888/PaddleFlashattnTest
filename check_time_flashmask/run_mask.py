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
    time_it,
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


def run_mask(
    run_mode,
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
):
    dtype = "float16"
    dropout = 0.0
    # test dynamic
    paddle.disable_static()

    q, k, v = genqkv(bsz, sq, sk, hq, hk, hdim, dtype)
    ograd = genograd(bsz, sq, hq, hdim, dtype)
    if mask_gen == "random":
        startend_row_indices = gen_random_flashmask(bsz, hm, sq, has_end, causal)
    elif mask_gen == "qamask":
        startend_row_indices = gen_qa_mask(bsz, hm, sq, has_end, causal)
    else:
        raise ValueError(f"{mask_gen} is not a valid value")
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
    return time_it(100, 10, fn)


if __name__ == "__main__":
    parser = get_fa_argparser()
    parser.add_argument("run_mode", type=str, choices=["flashmask"])
    parser.add_argument("case_id", type=int)
    parser.add_argument("device", type=str)
    parser.add_argument("log_file", type=str)
    args = parser.parse_args()
    paddle.set_device(args.device)
    run_mode = args.run_mode
    from cases import cases

    case = cases[args.case_id]

    with open(log_file, "a") as f:
        run_mask(*case)
        print(run_mode, *case, flush=True, file=f)
