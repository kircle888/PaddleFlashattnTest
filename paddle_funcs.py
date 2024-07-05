import paddle
from einops import rearrange, repeat
import time
import argparse
import numpy as np
import paddle.nn.functional as F


def clone_tensors(xs):
    ys = [x.detach().clone() for x in xs]
    for y in ys:
        y.stop_gradient = False
    return ys


def genqkv(bsz, sq, sk, hq, hk, hdim, dtype):
    shape_q = (bsz, sq, hq, hdim)
    shape_k = (bsz, sk, hk, hdim)
    q = paddle.randn(shape_q, dtype=dtype)
    k = paddle.randn(shape_k, dtype=dtype)
    v = paddle.randn(shape_k, dtype=dtype)
    q.stop_gradient = False
    k.stop_gradient = False
    v.stop_gradient = False
    return q, k, v


def genograd(bsz, sq, hq, hdim, dtype):
    shape_q = (bsz, sq, hq, hdim)
    ograd = paddle.randn(shape_q, dtype=dtype)
    return ograd


def gen_varlenqkv(bsz, sq, hq, hk, hdim, dtype):
    """
    Generate random varlen qkv.
    Returns:
        q, k, v: random varlen qkv.
        random_lens: the length of each sequence in a batch.
        cu_seqlen_q: cumsum of random_lens.
        key_padding_mask: mask for padding.
    """
    random_lens = np.random.randint(low=64, high=sq, size=(bsz), dtype=np.int32)
    cu_seqlen_q = np.cumsum(random_lens)
    cu_seqlen_q = np.concatenate([np.array([0]), cu_seqlen_q], axis=0)
    key_padding_mask = paddle.zeros((bsz, sq), dtype="int32")
    for i in range(bsz):
        key_padding_mask[i, : random_lens[i]] = 1
    nnz_q = cu_seqlen_q[-1]
    shape_q = (nnz_q, hq, hdim)
    shape_k = (nnz_q, hk, hdim)
    q = paddle.randn(shape_q, dtype=dtype)
    k = paddle.randn(shape_k, dtype=dtype)
    v = paddle.randn(shape_k, dtype=dtype)
    q.stop_gradient = False
    k.stop_gradient = False
    v.stop_gradient = False
    cu_seqlen_q = paddle.to_tensor(cu_seqlen_q, dtype="int32")
    return q, k, v, random_lens, cu_seqlen_q, key_padding_mask


def gen_varlenograd(shapeq, dtype):
    return paddle.randn(shapeq, dtype=dtype)


def packqkv(q, k, v):
    if len(q.shape) == 4:
        bsz, sq, hq, hdim = q.shape
        bsz, sk, hk, hdim = k.shape
        ng = hq // hk
        tq = q.reshape((0, 0, ng, nk, hdim))
        kv = paddle.stack([k, v], axis=2)
        qkv = paddle.concat([tq, kv], axis=2)
        return qkv
    elif len(q.shape) == 3:
        sq, hq, hdim = q.shape
        sk, hk, hdim = k.shape
        ng = hq // hk
        tq = q.reshape((0, ng, hk, hdim))
        kv = paddle.stack([k, v], axis=1)
        qkv = paddle.concat([tq, kv], axis=1)
        return qkv
    raise ValueError(f"q.shape: {q.shape}")


def unpackqkv(qkv):
    if len(qkv.shape) == 5:
        bsz, sq, ng, hk, hdim = qkv.shape
        qkv = paddle.flatten(qkv, 2, 3)
        q, k, v = paddle.split(
            qkv,
            num_or_sections=[
                ng * hk,
                hk,
                hk,
            ],
            axis=1,
        )
        return q, k, v
    elif len(qkv.shape) == 4:
        sq, ng, hk, hdim = qkv.shape
        qkv = paddle.flatten(qkv, 1, 2)
        q, k, v = paddle.split(
            qkv,
            num_or_sections=[
                ng * hk,
                hk,
                hk,
            ],
            axis=1,
        )
        return q, k, v
    raise ValueError(f"qkv.shape: {qkv.shape}")


class IndexFirstAxis(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = paddle.numel(paddle.to_tensor(other_shape))
        return paddle.gather(
            rearrange(input, "b ... -> b (...)"),
            repeat(indices, "z -> z d", d=second_dim),
            0,
        ).reshape((-1, *other_shape))

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensor()
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = paddle.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(indices, grad_output)
        return grad_input.reshape((ctx.first_axis_dim, *other_shape)), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = paddle.zeros((first_axis_dim, *values.shape[1:]), dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensor()
        grad_values = grad_output[indices]
        return grad_values, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype="int32")
    indices = paddle.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(paddle.cumsum(seqlens_in_batch, axis=0, dtype="int32"), (1, 0))
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def get_cuseqlens(key_padding_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        key_padding_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = key_padding_mask.sum(axis=-1, dtype="int32")
    max_seqlen_in_batch = key_padding_mask.shape[-1]
    cu_seqlens = F.pad(paddle.cumsum(seqlens_in_batch, axis=0, dtype="int32"), (1, 0))
    return (
        cu_seqlens,
        max_seqlen_in_batch,
    )


def scan_maxmin(arr, N):
    b, h, n = arr.shape
    d = (n + N - 1) // N
    max_arr = np.zeros((b, h, d))
    min_arr = np.zeros((b, h, d))
    for i in range(d):
        max_arr[:, :, i] = arr[:, :, (i * N) : (i + 1) * N].max(axis=2)
        min_arr[:, :, i] = arr[:, :, (i * N) : (i + 1) * N].min(axis=2)
    return max_arr, min_arr


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[-1]
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def pad(x, bsz, seqlen, cu_seqlen):
    rt = paddle.zeros((bsz, seqlen, *x.shape[1:]), dtype=x.dtype)
    for i in range(0, bsz):
        start = cu_seqlen[i]
        end = cu_seqlen[i + 1]
        lens = end - start
        rt[i, :lens] = x[start:end]
    return rt


def generate_startrows_padding(key_padding_mask):
    bsz = key_padding_mask.shape[0]
    seqlens_in_batch = key_padding_mask.sum(axis=-1, dtype="int32")
    max_seqlen = key_padding_mask.shape[-1]
    start_rows = paddle.zeros((bsz, 1, max_seqlen), dtype="int32")
    for i in range(bsz):
        start_rows[i, 0, seqlens_in_batch[i]] = seqlens_in_batch[i]
    return start_rows


def generate_startrows_unpad(h, cu_seqlen):
    nnz = cu_seqlen[-1]
    start_rows = paddle.zeros((1, h, nnz), dtype="int32")
    for i in range(cu_seqlen.shape[0] - 1):
        start_rows[:, :, cu_seqlen[i] : cu_seqlen[i + 1]] = cu_seqlen[i + 1]
    return start_rows


def cuseqlen_to_startrows(cu_seqlen, max_s):
    start_rows = np.zeros((max_s), dtype="int32")
    for i in range(cu_seqlen.shape[0] - 1):
        start_rows[cu_seqlen[i] : cu_seqlen[i + 1]] = cu_seqlen[i + 1]
    return start_rows


def generate_endrows_nomask(bz, num_head, max_k):
    end_rows = np.zeros((bz, num_head, max_k))
    return paddle.to_tensor(end_rows, dtype="int32")


def generate_endrows_causal(bz, num_head, max_k):
    end_rows = paddle.zeros((bz, num_head, max_k), dtype="int32")
    for i in range(max_k):
        end_rows[:, :, i] = i
    return end_rows


def generate_startrows_nomask(bz, seqlen):
    start_rows = np.ones((bz, 1, seqlen), dtype=int) * seqlen
    return paddle.to_tensor(start_rows, dtype="int32")


def generate_startrow_window(bz, num_head, seqlen):
    start_rows = np.zeros((bz, num_head, seqlen), dtype=int)
    for ik in range(seqlen):
        start_rows[:, :, ik] = min(seqlen, ik + 1024)
    return paddle.to_tensor(start_rows, dtype="int32")


def broadcast_startrows(start_rows, num_head, use_jump):
    start_rows = start_rows.numpy()
    assert num_head % start_rows.shape[1] == 0
    if not use_jump:
        ret = np.repeat(start_rows, num_head // start_rows.shape[1], axis=1)
    else:
        ret = np.expand_dims(start_rows, 1)
        ret = np.repeat(ret, num_head // start_rows.shape[1], axis=1)
        ret = ret.reshape((ret.shape[0], num_head, start_rows.shape[2]))
    return paddle.to_tensor(ret, dtype="int32")


def generate_mask_matrix_from_mask_indices(start_rows):
    bz, num_head, seq_len = start_rows.shape
    matrix = np.zeros((seq_len, seq_len))
    matrix[np.triu_indices(seq_len, 1)] = -np.inf
    matrix = matrix[np.newaxis, np.newaxis, :, :]
    matrix = np.tile(matrix, (bz, num_head, 1, 1))

    for bz_idx in range(bz):
        for head_idx in range(num_head):
            for j in range(seq_len):
                start_row = start_rows[bz_idx, head_idx, j]
                matrix[bz_idx, head_idx, start_row:, j] = -np.inf
                matrix[bz_idx, head_idx, j, j] = 0.0
    return matrix


def generate_mask_from_sparsemask(start_rows, end_rows, dtype, causal=True):
    bz, num_head, seq_len = start_rows.shape
    m = paddle.zeros((bz, num_head, seq_len, seq_len), dtype=dtype)
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seq_len):
                start = start_rows[bi, hi, j]
                m[bi, hi, start:, j] = -np.inf
                if end_rows is not None:
                    end = end_rows[bi, hi, j]
                    m[bi, hi, :end, j] = -np.inf
                if causal:
                    m[bi, hi, :j, j] = -np.inf
    return m


def flashmask_to_densemask(startend_row_indices, dtype, causal=True):
    bz, num_head, seq_len, bound_num = startend_row_indices.shape
    m = paddle.zeros((bz, num_head, seq_len, seq_len), dtype=dtype)
    has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seq_len):
                downstart = startend_row_indices[bi, hi, j, 0]
                if has_end:
                    downend = startend_row_indices[bi, hi, j, 1]
                    m[bi, hi, downstart:downend, j] = -np.inf
                else:
                    m[bi, hi, downstart:, j] = -np.inf
                if causal:
                    m[bi, hi, :j, j] = -np.inf
                else:
                    if has_end:
                        upstart = startend_row_indices[bi, hi, j, 2]
                        upend = startend_row_indices[bi, hi, j, 3]
                        m[bi, hi, upstart:upend, j] = -np.inf
                    else:
                        upend = startend_row_indices[bi, hi, j, 1]
                        m[bi, hi, :upend, j] = -np.inf
    return m


def generate_random_block_mask(bz, num_head, seqlen, causal=True):
    start_rows = paddle.zeros((bz, num_head, seqlen), dtype="int32")
    if not causal:
        end_rows = paddle.zeros((bz, num_head, seqlen), dtype="int32")
    for ib in range(bz):
        for ih in range(num_head):
            split_num = np.random.randint(5, 10)
            split_points = np.random.randint(1, seqlen, size=split_num)
            cu_seqlens = list(np.sort(np.unique(split_points)))
            cu_seqlens.insert(0, 0)
            cu_seqlens.append(seqlen)
            split_num = len(cu_seqlens) - 1
            for i in range(split_num):
                start_rows[ib, ih, cu_seqlens[i] : cu_seqlens[i + 1]] = int(
                    cu_seqlens[i + 1]
                )
                if not causal:
                    end_rows[ib, ih, cu_seqlens[i] : cu_seqlens[i + 1]] = int(
                        cu_seqlens[i]
                    )
    if not causal:
        return start_rows, end_rows
    else:
        return start_rows, None


def gen_random_block_dataset(bz, seqlen, splitnum):
    split_points = np.random.randint(128, seqlen, size=(bz, splitnum), dtype=np.int32)
    cu_seqlens = np.concatenate(
        [np.zeros((bz, 1), dtype=np.int32), split_points], axis=1
    )
    cu_seqlens = np.sort(cu_seqlens, axis=1)
    seqlens = np.diff(cu_seqlens, axis=1).astype(np.int64)
    densities = np.sum(seqlens**2, axis=1) / (seqlen**2)
    sparsities = 1 - densities
    if not (np.all(sparsities <= 1) and np.all(sparsities >= 0)):
        import pdb

        pdb.set_trace()
    return cu_seqlens, sparsities


def gen_random_sparsemask(bz, seqlen):
    return paddle.randint(0, seqlen, shape=(bz, 1, seqlen), dtype="int32")


def generate_start_rows(bz, num_head, rows, cols, start_row):
    assert rows == cols, f"rows {rows} must be equal to cols {cols}."
    start_rows_list = []
    for bz_idx in range(bz):
        for head_idx in range(num_head):
            start_rows = np.array([rows + 1] * cols)
            mask_pos = np.random.choice(cols - 1, cols - start_row, replace=False)
            index = np.arange(start_row, rows)
            mask_pos = np.concatenate(
                [mask_pos[mask_pos < index - 1], mask_pos[mask_pos >= index - 1]]
            )
            start_rows[mask_pos] = index
            start_rows_list.append(start_rows)
    start_rows_arr = np.array(start_rows_list).reshape([bz, num_head, rows])
    return start_rows_arr


def gen_sparsemask_fromqalens(qalens, max_s, causal=True, has_end=False):
    assert causal or not has_end, "causal and end not support"
    mask_dim = 1
    if not causal:
        mask_dim *= 2
    if has_end:
        mask_dim *= 2
    start_rows = np.zeros((max_s, mask_dim), dtype="int32")
    cu_seqlen = 0
    for qalen in qalens:
        query = True
        seqlen = sum(qalen)
        for vlen in qalen:
            if query:
                v = cu_seqlen + seqlen
                query = False
            else:
                v = cu_seqlen + vlen
            start_rows[cu_seqlen : cu_seqlen + vlen, 0] = v
            if not causal:
                start_rows[cu_seqlen : cu_seqlen + vlen, 1] = cu_seqlen
            assert v < max_s
            cu_seqlen += vlen
    start_rows[cu_seqlen:, 0] = max_s
    if has_end:
        ending_pos = np.random.randint(0, seqlen)
        start_rows[:, 1] = ending_pos
        start_rows[:, 1] = np.maximum(start_rows[:, 0], start_rows[:, 1])
    return start_rows


def gen_query_multians_dataset(bz, seqlen, splitnum, ansnum):
    ret = []
    min_len = 16
    while len(ret) < bz:
        split_point = np.random.randint(
            min_len, seqlen, size=splitnum, dtype=np.int32
        ).tolist()
        split_point.append(0)
        split_point.sort()
        cu_seqlen = split_point
        if cu_seqlen[-1] + min_len <= seqlen:
            continue
        seqlens = np.diff(cu_seqlen)
        seqlens = seqlens.tolist()
        queryanslens = []
        for i in seqlens:
            ansmin_factor = 0.10 / (1 + 0.10 * ansnum)
            ansmax_factor = 0.20 / (1 + 0.20 * ansnum)
            anslen = np.random.randint(
                int(i * ansmin_factor), int(i * ansmax_factor) + 1, size=ansnum
            ).tolist()
            querylen = i - sum(anslen)
            queryanslens.append([querylen, *anslen])
        ret.append(queryanslens)
    return ret


def gen_random_flashmask(bz, num_head, seqlen, has_end, causal):
    mask_num = 1
    if not causal:
        mask_num *= 2
    if has_end:
        mask_num *= 2
    m = np.random.randint(0, seqlen, (bz, num_head, seqlen, mask_num))
    diag = np.arange(seqlen).reshape((1, 1, seqlen))
    m[:, :, :, 0] = np.maximum(diag + 1, m[:, :, :, 0])
    if not causal:
        if has_end:
            raise NotImplementedError()
        else:
            m[:, :, :, 1] = np.minimum(diag, m[:, :, :, 1])
    else:
        if has_end:
            m[:, :, :, 1] = m[:, :, :, 0] + 1
            m[:, :, :, 1] = np.maximum(m[:, :, :, 0], m[:, :, :, 1])

    return paddle.to_tensor(m, dtype="int32")


def gen_qa_mask(bz, num_head, seqlen, has_end, causal):
    mask_dim = 1
    if not causal:
        mask_dim *= 2
    if has_end:
        mask_dim *= 2
    spnum = np.random.randint(1, 10)
    ansnum = np.random.randint(1, 7)
    batch_qalens = gen_query_multians_dataset(bz * num_head, seqlen, spnum, ansnum)
    batch_qalens = [
        gen_sparsemask_fromqalens(x, seqlen, causal, has_end) for x in batch_qalens
    ]
    ret = np.stack(batch_qalens, axis=0)
    ret = np.reshape(ret, (bz, num_head, seqlen, mask_dim))
    return paddle.to_tensor(ret, dtype="int32")


def calc_sparsity(start_rows, max_s):
    cons = np.expand_dims(np.arange(0, max_s, dtype=np.int64), axis=0)
    start_rows = np.maximum(start_rows, cons)
    masked_len = (max_s - start_rows).astype(np.int64)
    sum_len = np.sum(masked_len, axis=1)
    sparsity = sum_len / (max_s * (max_s + 1) / 2)
    if not np.all(sparsity <= 1) or not np.all(sparsity >= 0):
        import pdb

        pdb.set_trace()
    return sparsity


def calc_block_sparsity(start_rows, max_s, block_m, block_n):
    bs_rows, _ = scan_maxmin(np.expand_dims(start_rows, axis=0), block_n)
    max_m_block = (max_s + block_m - 1) // block_m
    bs_rows = np.ceil(bs_rows / block_m)
    masked_block = max_m_block - bs_rows
    sum_block = np.sum(masked_block)
    sparsity = sum_block / (max_m_block * (max_m_block + 1) / 2)
    return sparsity


def generate_intokens_mask(bz, max_s):
    start_rows = paddle.zeros((bz, 1, max_s), dtype="int32")
    inbatchs = []
    for ib in range(bz):
        cu_seqlen = 0
        inbatch_seqlen = []
        while True:
            seqlen = np.random.randint(128, max_s)
            next_cu_seqlen = cu_seqlen + seqlen
            if next_cu_seqlen < max_s:
                start_rows[ib, :, cu_seqlen:next_cu_seqlen] = next_cu_seqlen
                inbatch_seqlen.append(seqlen)
                cu_seqlen = next_cu_seqlen
            else:
                break
        inbatchs.append(inbatch_seqlen)
    return start_rows, inbatchs


def check(x, y):
    if isinstance(x, paddle.Tensor):
        x = x.cast("float32").numpy()
    if isinstance(y, paddle.Tensor):
        y = y.cast("float32").numpy()
    np.testing.assert_allclose(x, y, rtol=1e-02, atol=1e-02)


def strict_check(x, y):
    if isinstance(x, paddle.Tensor):
        x = x.cast("float32").numpy()
    if isinstance(y, paddle.Tensor):
        y = y.cast("float32").numpy()
    np.testing.assert_array_equal(x, y)


def time_paddle():
    paddle.device.synchronize()
    return time.time()


def time_it(run_iter: int, warmup_iter: int, target_func):
    for idx in range(run_iter + warmup_iter):
        if idx == warmup_iter:
            sta_time = time_paddle()
        target_func()
    end_time = time_paddle()
    return (end_time - sta_time) / run_iter


def get_fa_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checking", action="store_true", help="whether to run in checking mode"
    )
    parser.add_argument(
        "--time_test", action="store_true", help="whether to run in time test mode"
    )
    parser.add_argument(
        "--h", type=str, default="8,8,8", help="number of heads,format: hq,hk,hm"
    )
    parser.add_argument("--seed", type=int, default=68, help="random seed")
    parser.add_argument("--lq", type=int, default=16 * 1024, help="seqlen of q")
    parser.add_argument("--lk", type=int, default=16 * 1024, help="seqlen of k")
    parser.add_argument("--bsz", type=int, default=16, help="batch size")
    parser.add_argument("--hdim", type=int, default=128, help="head dimension")
    parser.add_argument(
        "--non_causal", action="store_true", help="whether to run not causal"
    )
    parser.add_argument("--log_file", type=str, default="sparsefa.log")
    return parser


def parse_h(args):
    hs = args.h.split(",")
    hq = int(hs[0])
    hk = int(hs[1])
    hm = int(hs[2])
    return hq, hk, hm


def print_config(args, name, file):
    hq, hk, hm = parse_h(args)
    print(
        name,
        args.bsz,
        args.lq,
        args.lk,
        hq,
        hk,
        hm,
        args.hdim,
        # args.sparsity,
        args.seed,
        # args.start_rows_file,
        file=file,
        flush=True,
    )


class PaddleRandomizer:
    def __init__(self):
        pass

    def seed(self, seed):
        paddle.seed(seed)

    def __call__(self, shape, dtype):
        return paddle.randn(shape, dtype=dtype)


class PaddleTensorizer:
    def __init__(self):
        pass

    def __call__(self, nparray, dtype):
        return paddle.to_tensor(nparray, dtype=dtype)
