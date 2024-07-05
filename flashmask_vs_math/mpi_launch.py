from flashmask_vs_math import test_sparsefa
import itertools
import os
import paddle

if __name__ == "__main__":
    testcases = []
    batchszs = [1]
    headdims = [32, 128]
    seeds = [0]
    seqlens = [128, 256, 512, 1024, 2048, 4096, 8192]
    seqlens.extend([2394, 4072, 3738, 3957, 1492, 857, 8999, 6148, 2510, 5071])
    hs = [(8, 8, 8)]
    flags = [(True, True), (True, False), (False, False)]
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
    worldsize = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    gpu = f"gpu:{rank % 8}"
    part_size = (len(testcases) + worldsize - 1) // worldsize
    start = rank * part_size
    end = (rank + 1) * part_size
    cases = testcases[start:end]
    paddle.set_device(gpu)
    for case in cases:
        case.append(f"./log/rank{rank}.log")
        test_sparsefa(*case)
