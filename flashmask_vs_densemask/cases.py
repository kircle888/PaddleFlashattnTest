import itertools


def gen_cases():
    testcases = []
    batchszs = [1]
    headdims = [32, 128]
    seeds = [0]
    seqlens = [128, 256, 512, 1024, 2048, 4096, 8192]
    seqlens.extend([2394, 4072, 3738, 3957, 1492, 857, 8999, 6148, 2510, 5071])
    hs = [(8, 8, 8)]
    # flags = [(True, True), (True, False), (False, False)]
    flags = [(True, False), (False, False)]
    mask_gen = ["random", "qamask"]
    for i in itertools.product(batchszs, flags, seeds, hs, headdims, seqlens, mask_gen):
        case = [
            i[0],
            i[5],
            i[5],
            i[3][0],
            i[3][1],
            i[3][2],
            i[4],
            i[1][0],
            i[1][1],
            i[6],
        ]
        testcases.append(case)
    return testcases


cases = gen_cases()

if __name__ == "__main__":
    print(len(cases))
