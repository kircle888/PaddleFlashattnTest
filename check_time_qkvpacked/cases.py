import itertools


def gen_cases():
    testcases = []
    batchszs = [1, 4, 8]
    headdims = [32, 128]
    seqlens = [1024, 2048, 4096, 8192, 16 * 1024, 32 * 1024]
    # seqlens.extend([2394, 4072, 3738, 3957, 1492, 857, 8999, 6148, 2510, 5071])
    hs = [(8, 8), (32, 32)]
    # hs = [(96,8)]
    causal = [True, False]
    dtypes = ["bfloat16"]
    for i in itertools.product(batchszs, hs, headdims, seqlens, causal, dtypes):
        case = [i[0], i[3], i[3], i[1][0], i[1][1], i[2], i[4], i[5]]
        testcases.append(case)
    return testcases


cases = gen_cases()

if __name__ == "__main__":
    print(len(cases))
