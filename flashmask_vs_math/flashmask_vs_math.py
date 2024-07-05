import paddle
import paddle.nn.functional as F
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
import numpy as np


def attention_naive_with_mask(q, k, v, attn_bias):
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)
    p = F.softmax(s + attn_bias)
    o = paddle.matmul(p, vt)
    return paddle.transpose(o, [0, 2, 1, 3])


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
    if run_mode == "attentionmask":
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

    elif run_mode == "math_attention":
        attn_mask = flashmask_to_densemask(
            startend_row_indices,
            "float32",
            causal=causal,
        )

        def target_func():
            q1, k1, v1 = q.cast("float32"), k.cast("float32"), v.cast("float32")
            ograd1 = ograd.cast("float32")
            q1, k1, v1 = clone_tensors([q1, k1, v1])
            out = attention_naive_with_mask(q1, k1, v1, attn_mask)
            out.backward(ograd1)
            return out, q1.grad, k1.grad, v1.grad

    elif run_mode == "old_sparsemask":
        assert causal
        start_row_indices = startend_row_indices.unsqueeze(-1)

        def target_func():
            q1, k1, v1 = clone_tensors([q, k, v])
            out = F.flash_attention_with_sparse_mask(
                q1,
                k1,
                v1,
                attn_mask_start_row_indices=start_row_indices,
                attn_mask_start_row=0,
                dropout_p=dropout,
                is_causal=True,
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


def test_sparsefa(
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
        "flashmask",
        q,
        k,
        v,
        startend_row_indices=startend_row_indices,
        dropout=dropout,
        causal=causal,
        ograd=ograd,
    )
    out, qgrad, kgrad, vgrad = fn()

    stdfn = get_fafunc(
        "math_attention",
        q,
        k,
        v,
        startend_row_indices=startend_row_indices,
        dropout=dropout,
        causal=causal,
        ograd=ograd,
    )
    out_std, qgrad_std, kgrad_std, vgrad_std = stdfn()
    try:
        check(out, out_std)
        check(qgrad, qgrad_std)
        check(kgrad, kgrad_std)
        check(vgrad, vgrad_std)
    except AssertionError as e:
        with open(log_file, "a") as f:
            print("Test Failed", flush=True, file=f)
            print(e, flush=True, file=f)
        # import pdb

        # pdb.set_trace()
        # raise e
        return
    with open(log_file, "a") as f:
        print("Test Pass", flush=True, file=f)


if __name__ == "__main__":
    parser = get_fa_argparser()
    parser.add_argument("--device", type=str, default="gpu:0")
    parser.add_argument(
        "--mask_gen", type=str, default="random", choices=["random", "qamask"]
    )
    args = parser.parse_args()
    paddle.set_device(args.device)
    log_file = args.log_file
    paddle.seed(args.seed)
    np.random.seed(args.seed)
    hq, hk, hm = parse_h(args)
    test_sparsefa(
        args.bsz,
        args.lq,
        args.lk,
        hq,
        hk,
        hm,
        args.hdim,
        not args.non_causal,
        True,
        args.mask_gen,
        log_file,
    )
