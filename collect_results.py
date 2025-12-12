import re, csv, sys
from pathlib import Path

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs")
OUT  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results_summary.csv")

# Patterns expected from your logs
epoch_line = re.compile(r"^(\d+)\s+\t\s+[0-9.]+\s+\t\s+[0-9.eE+-]+\s+\t\s+([0-9.eE+-]+)\s+\t\s+([0-9.eE+-]+)")
test_header = re.compile(r"Test Loss\s+\t\s+Test Acc\s+\t\s+PGD Loss\s+\t\s+PGD Acc")
test_line = re.compile(r"([0-9.]+)\s+\t\s+\t\s+([0-9.]+)\s+\t\s+([0-9.]+)\s+\t\s+([0-9.]+)")

rows = []
for log in BASE.rglob("output.log"):
    # infer params from path segments
    path = log.as_posix()
    def grab(tag):
        m = re.search(fr"{tag}_(\d+(?:\.\d+)?)", path)
        return m.group(1) if m else ""
    beta = grab("beta")
    lam  = grab("lamda")
    lsc  = grab("scale|lscale")  # accommodates our out_dir and your internal directory
    lstart = grab("time") or "50"
    seed_m = re.search(r"seed_(\d+)", path)
    seed = seed_m.group(1) if seed_m else ""

    test_acc = pgd_acc = test_loss = pgd_loss = ""
    with open(log, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        # Capture the last occurrence of the test metrics block
        for i, line in enumerate(lines):
            if test_header.search(line):
                if i + 1 < len(lines) and (m := test_line.search(lines[i + 1])):
                    test_loss, test_acc, pgd_loss, pgd_acc = m.groups()

    rows.append([beta, lam, lsc, lstart, seed, path, test_loss, test_acc, pgd_loss, pgd_acc])

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["beta", "lamda", "lam_scale", "lam_start", "seed", "log_path", "test_loss", "test_acc", "pgd_loss", "pgd_acc"])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT}")
