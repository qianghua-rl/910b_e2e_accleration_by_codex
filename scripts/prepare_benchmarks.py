"""Convert raw benchmark files into the jsonl format used by cb_run.py.

Output rows: {"id", "question", "answer"} (+ optional extra fields), one per line.
  - math500:  raw test.jsonl from HuggingFaceH4/MATH-500 (fields problem/answer)
  - aime2024: raw aime_2024.parquet from Maxwell-Jia/AIME_2024 (ID/Problem/Answer)
  - aime2025: raw aime_2025.parquet from MathArena/aime_2025 (problem_idx/problem/answer)

Usage:
  python prepare_benchmarks.py <raw_dir> <out_dir>
Expects <raw_dir>/math500/test.jsonl, <raw_dir>/aime24/aime_2024.parquet,
<raw_dir>/aime25/aime_2025.parquet. Missing inputs are skipped with a message.
Parquet is read with pyarrow only; no code inside the data files is executed.
"""
import hashlib
import json
import sys
from pathlib import Path


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    print(f"{path}: {len(rows)} rows, sha256[:16]={sha}")


def convert_math500(raw, out):
    rows = []
    for i, line in enumerate(open(raw, encoding="utf-8")):
        d = json.loads(line)
        rows.append({"id": i, "question": d["problem"], "answer": d["answer"].strip(),
                     "subject": d.get("subject"), "level": d.get("level"),
                     "unique_id": d.get("unique_id")})
    write_jsonl(out, rows)


def convert_aime24(raw, out):
    import pyarrow.parquet as pq
    rows = []
    for i, d in enumerate(pq.read_table(raw).to_pylist()):
        rows.append({"id": i, "question": d["Problem"], "answer": str(d["Answer"]).strip(),
                     "source_id": d["ID"]})
    write_jsonl(out, rows)


def convert_aime25(raw, out):
    import pyarrow.parquet as pq
    rows = []
    for d in pq.read_table(raw).to_pylist():
        rows.append({"id": int(d["problem_idx"]), "question": d["problem"],
                     "answer": str(d["answer"]).strip()})
    write_jsonl(out, rows)


def convert_gpqa(raw, out):
    import pyarrow.parquet as pq
    rows = []
    for i, d in enumerate(pq.read_table(raw).to_pylist()):
        rows.append({"id": i, "question": d["question"].strip(), "answer": str(d["answer"]).strip()})
    write_jsonl(out, rows)


BBH_PER_TASK = 50   # subsample per BBH task (27 tasks, 6511 total); fixed seed for reproducibility


def convert_bbh(raw_root, out):
    """raw_root: directory holding <task>/test-00000-of-00001.parquet (lukaemon/bbh layout)."""
    import random
    import pyarrow.parquet as pq
    rows = []
    rng = random.Random(0)
    for task_dir in sorted(p for p in Path(raw_root).iterdir() if p.is_dir()):
        items = pq.read_table(task_dir / "test-00000-of-00001.parquet").to_pylist()
        idx = sorted(rng.sample(range(len(items)), min(BBH_PER_TASK, len(items))))
        for j in idx:
            d = items[j]
            rows.append({"id": f"{task_dir.name}/{j}", "task": task_dir.name,
                         "question": d["input"], "answer": str(d["target"]).strip()})
    write_jsonl(out, rows)


JOBS = [
    ("math500/test.jsonl", "math500.jsonl", convert_math500),
    ("aime24/aime_2024.parquet", "aime2024.jsonl", convert_aime24),
    ("aime25/aime_2025.parquet", "aime2025.jsonl", convert_aime25),
    ("gpqa/gpqa_diamond.parquet", "gpqa_diamond.jsonl", convert_gpqa),
    ("bbh", "bbh.jsonl", convert_bbh),
]


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    raw_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel, out_name, fn in JOBS:
        src = raw_dir / rel
        if not src.exists():
            print(f"skip: {src} not found")
            continue
        fn(src, out_dir / out_name)


if __name__ == "__main__":
    main()
