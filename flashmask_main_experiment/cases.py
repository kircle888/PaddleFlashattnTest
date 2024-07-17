import itertools


def gen_cases():
    testcases = []
    batchszs = [1]
    headdims = [128]
    seqlens = [2048, 4096, 8192, 16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024]
    # seqlens = [64 * 1024]
    hs = [(32, 32)]
    causal = [True]
    has_end = [False]
    sparsity = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # qaratio = [1, 2, 6]
    qaratio = [1]
    for i in itertools.product(
        batchszs, hs, headdims, qaratio, sparsity, seqlens, causal, has_end
    ):
        case = [i[0], i[5], i[5], i[1][0], i[1][1], i[2], i[4], i[3], i[6], i[7]]
        testcases.append(case)
    return testcases


cases = gen_cases()

if __name__ == "__main__":
    print(len(cases))
