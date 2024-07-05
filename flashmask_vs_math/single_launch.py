from flashmask_vs_math import test_sparsefa
import itertools
import os
import paddle

if __name__ == "__main__":
    testcases = [[1, 1024, 1024, 32, 32, 32, 96, True, False, "random"]]
    for case in testcases:
        case.append(f"./log/checking.log")
        test_sparsefa(*case)
