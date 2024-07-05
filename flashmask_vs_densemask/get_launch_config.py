import itertools
import os
import paddle
from cases import cases
from pathlib import Path
import sys

if __name__ == "__main__":
    case_num = len(cases)
    if os.environ.get("OMPI_COMM_WORLD_SIZE", None) is not None:
        worldsize = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        gpu = f"gpu:{rank % 8}"
        part_size = (case_num + worldsize - 1) // worldsize
        start = rank * part_size
        end = (rank + 1) * part_size
        mycases = range(start, end)
        mycases = [str(x) for x in mycases]
        paddle.set_device(gpu)
        log_file = Path(f"./logs/rank{rank}.log").resolve()
    else:
        mycases = range(case_num)
        mycases = [str(x) for x in mycases]
        log_file = Path(f"./logs/checking.log").resolve()
        gpu = "gpu:1"
    arg_v = int(sys.argv[1])
    if arg_v == 0:
        print(" ".join(mycases))
    elif arg_v == 1:
        print(gpu)
    else:
        print(str(log_file))
