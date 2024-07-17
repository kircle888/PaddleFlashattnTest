import itertools


def gen_cases():
    testcases = []
    headdims = [128]
    # seqlens = [1024, 2048, 4096, 8192, 16 * 1024, 32 * 1024]
    seqlens = [
        2048,
        4096,
        8192,
        16 * 1024,
        32 * 1024,
        64 * 1024,
        128 * 1024,
        256 * 1024,
    ]
    seqlens = [16 * 1024]
    # seqlens.extend([2394, 4072, 3738, 3957, 1492, 857, 8999, 6148, 2510, 5071])
    hs = [(8, 8)]
    # hs = [(96,8), (32, 32)]
    causal = [True]
    dtypes = ["bfloat16"]
    # qaratios = [1, 2, 6]
    qaratios = [2]
    for i in itertools.product(hs, headdims, seqlens, causal, dtypes, qaratios):
        case = [i[2], i[2], i[0][0], i[0][1], i[1], i[3], i[4], i[5]]
        testcases.append(case)
    cases = [[]]
    return testcases


cases = gen_cases()

if __name__ == "__main__":
    print(len(cases))
