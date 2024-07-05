import tabulate


def parse_log(log_file):
    with open(log_file, "r") as f:
        lines = f.readlines()
    title = None
    v = dict()
    for line in lines:
        if title is None:
            title = line
        else:
            v[title] = float(line)
            title = None
    return v


if __name__ == "__main__":
    std_d = parse_log("std_fa.log")
    dev_d = parse_log("dev_fa.log")
    table = []
    for case, value in std_d.items():
        dev_value = dev_d[case]
        diff = (dev_value - value) / value
        table.append([case, value, dev_value, f"{diff :.2%}"])
    print(tabulate.tabulate(table, headers=["Case", "Standard", "Dev", "Diff"]))
