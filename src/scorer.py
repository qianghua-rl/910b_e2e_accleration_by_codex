"""Frozen answer scorer for the fresh workspace.

Version: fresh-v1. Changing any rule below requires a new version string and a
disclosed re-score of all prior results. Do not tune rules against candidate outputs.

Rules:
  - prediction = content of the LAST \\boxed{...} in the final assistant text
  - gold = the `answer` field of the jsonl (a number as a string)
  - both are parsed to float after removing commas, dollar signs and spaces
  - if the prediction is not a clean number, the last number in it is used
  - correct iff |pred - gold| < 1e-6
Usage: python scorer.py <per_task.jsonl>   (re-scores and prints a summary, CPU only)
"""
import json
import re
import sys

SCORER_VERSION = "fresh-v1"
TOL = 1e-6


def extract_boxed(text):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text or "")
    return m[-1].strip() if m else None


def to_number(s):
    if s is None:
        return None
    s = str(s).replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        return float(nums[-1]) if nums else None


def score_one(final_text, gold):
    pred = extract_boxed(final_text)
    p = to_number(pred)
    g = to_number(gold)
    correct = p is not None and g is not None and abs(p - g) < TOL
    return {"pred": pred, "gold": gold, "correct": correct}


def score_file(path):
    rows = [json.loads(l) for l in open(path)]
    out = []
    for r in rows:
        s = score_one(r.get("final_text", ""), r["gold"])
        out.append({"id": r["id"], **s})
    n_correct = sum(x["correct"] for x in out)
    return out, n_correct, len(out)


def _selftest():
    cases = [
        ("so \\boxed{8}", "8", True),
        ("first \\boxed{3} then \\boxed{8}", "8", True),
        ("\\boxed{1,000}", "1000", True),
        ("\\boxed{\\$5.00}", "5", True),
        ("the answer is 5", "5", False),          # no boxed -> wrong
        ("\\boxed{7}", "8", False),
        ("\\boxed{}", "8", False),
        ("\\boxed{about 36 apples}", "36", True),  # last number fallback
    ]
    for text, gold, want in cases:
        got = score_one(text, gold)["correct"]
        assert got == want, (text, gold, got, want)
    print("scorer selftest ok:", SCORER_VERSION)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _selftest()
    else:
        out, c, n = score_file(sys.argv[1])
        print(f"{SCORER_VERSION}: {c}/{n} correct")
        for x in out:
            if not x["correct"]:
                print("wrong:", x)
