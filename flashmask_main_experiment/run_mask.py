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
):
    if run_mode == "densemask":
        attn_mask = flashmask_to_densemask(
            startend_row_indices,
            q.dtype,
            causal=causal,
        )
        q1, k1, v1 = clone_tensors([q, k, v])

        def target_func():
            out = F.scaled_dot_product_attention(
                q1,
                k1,
                v1,
                attn_mask=attn_mask,
                is_causal=False,
            )
            out.backward(retain_graph=True)
            return out, q1.grad, k1.grad, v1.grad

    elif run_mode == "flashmask":
        q1, k1, v1 = clone_tensors([q, k, v])

        def target_func():
            out = F.flashmask_attention(
                q1,
                k1,
                v1,
                startend_row_indices=startend_row_indices,
                dropout=dropout,
                causal=causal,
            )
            out.backward(retain_graph=True)
            return out, q1.grad, k1.grad, v1.grad

    else:
        assert 0
    return target_func


def load_inputs(bsz, sq, sk, hq, hk, hdim, dtype, qaratio, sparsity):
    filename = f"data/srowsQ{qaratio}A{sq//1024}K.npz"
    d = np.load(filename)
    if str(sparsity) not in d:
        exit(0)
    d = d[str(sparsity)]
    # np.random.shuffle(d)
    # d = d[:200]
    for item in d:
        q, k, v = genqkv(bsz, sq, sk, hq, hk, hdim, dtype)
        startend_rows = (
            paddle.to_tensor(item, dtype="int32")
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        yield q, k, v, startend_rows


def run_mask(
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
    assert causal == True
    assert has_end == False
    dtype = "float16"
    dropout = 0.0
    # test dynamic
    paddle.disable_static()

    gener = load_inputs(bsz, sq, sk, hq, hk, hdim, dtype, qaratio, sparsity)

    times = []
    for q, k, v, startend_row_indices in gener:
        fn = get_fafunc(
            run_mode,
            q,
            k,
            v,
            startend_row_indices=startend_row_indices,
            dropout=dropout,
            causal=causal,
        )
        t = time_it(50, 5, fn)
        times.append(t)
    return np.average(times)


if __name__ == "__main__":
    parser = get_fa_argparser()
    parser.add_argument("run_mode", type=str, choices=["flashmask", "densemask"])
    parser.add_argument("case_id", type=int)
    parser.add_argument("device", type=str)
    parser.add_argument("log_file", type=str)
    args = parser.parse_args()
    paddle.set_device(args.device)
    run_mode = args.run_mode
    from cases import cases

    case = cases[args.case_id]

    with open(args.log_file, "a") as f:
        t = run_mask(run_mode, *case)
        print(run_mode, *case, flush=True, file=f)
        print(t, flush=True, file=f)
