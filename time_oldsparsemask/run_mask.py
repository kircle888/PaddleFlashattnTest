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
    gen_sparsemask_fromqalens,
    time_it,
)
from pathlib import Path
import json
import numpy as np

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

        def target_func():
            q1, k1, v1 = clone_tensors([q, k, v])
            out = F.scaled_dot_product_attention(
                q1,
                k1,
                v1,
                attn_mask=attn_mask,
                is_causal=False,
            )
            out.backward()
            return out, q1.grad, k1.grad, v1.grad

    elif run_mode == "oldsparsemask":
        q1, k1, v1 = clone_tensors([q, k, v])
        assert startend_row_indices.shape[-1] == 1
        assert causal == True

        def target_func():
            out = F.flash_attention_with_sparse_mask(
                q1, k1, v1, startend_row_indices.squeeze(-1), is_causal=True
            )
            out.backward()
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
            out.backward()
            return out, q1.grad, k1.grad, v1.grad

    else:
        assert 0
    return target_func


def get_inputs(sq, sk, hq, hk, hdim, dtype, qaratio):
    if sq < 1024:
        ks = sq / 1024
        ks = f"{ks :2}"
    else:
        ks = sq // 1024
        ks = str(ks)
    with Path(".", "data", f"metadataQ{qaratio}A{ks}K.json").open("r") as f:
        d = json.load(f)
    np.random.shuffle(d)
    d = d[:200]
    for item in d:
        q, k, v = genqkv(1, sq, sk, hq, hk, hdim, dtype)
        startend_rows = gen_sparsemask_fromqalens(item, sq).unsqueeze(0).unsqueeze(0)
        yield q, k, v, startend_rows


def run_mask(run_mode, sq, sk, hq, hk, hdim, causal, dtype, qaratio):
    assert sq == sk
    assert causal == True

    times = []
    gener = get_inputs(sq, sk, hq, hk, hdim, dtype, qaratio)
    for q, k, v, startend_row_indices in gener:

        fn = get_fafunc(
            run_mode,
            q,
            k,
            v,
            startend_row_indices=startend_row_indices,
            dropout=0.0,
            causal=causal,
        )

        t = time_it(100, 10, fn)
        times.append(t)
    return np.average(times)


if __name__ == "__main__":
    parser = get_fa_argparser()
    parser.add_argument("run_mode", type=str, choices=["flashmask", "oldsparsemask"])
    parser.add_argument("case_id", type=int)
    parser.add_argument("device", type=str)
    parser.add_argument("log_file", type=str)
    args = parser.parse_args()
    paddle.set_device(args.device)
    run_mode = args.run_mode
    from cases import cases

    case = cases[args.case_id]
    t = run_mask(run_mode, *case)

    with open(args.log_file, "a") as f:
        print(run_mode, *case, flush=True, file=f)
        print(t, flush=True, file=f)
