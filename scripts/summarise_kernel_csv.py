"""Summarise an Ascend profiler kernel_details.csv: total device time per op name (top N)."""
import csv
import glob
import sys
from collections import defaultdict

root = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
path = glob.glob(f"{root}/**/kernel_details.csv", recursive=True)[0]
tot = defaultdict(float)
cnt = defaultdict(int)
with open(path) as f:
    for r in csv.DictReader(f):
        name = r.get("Name") or r.get("Kernel Name") or "?"
        dur = float(r.get("Duration(us)") or r.get("Duration") or 0)
        tot[name] += dur
        cnt[name] += 1
total = sum(tot.values())
print(f"file={path} total_device_ms={total/1000:.1f}")
for name, d in sorted(tot.items(), key=lambda x: -x[1])[:top]:
    print(f"{d/1000:10.1f} ms {100*d/total:5.1f}%  n={cnt[name]:6d}  {name[:90]}")
