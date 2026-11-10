"""Offline scorer for multiple-choice / short-answer benchmarks (GPQA Diamond, BBH).

Version: fresh-mc-v1. Re-scores a per_task.jsonl produced by cb_run.py --direct.

Rules:
  - prediction = content of the LAST \\boxed{...} in final_text (balanced braces,
    via scorer_math.last_boxed).
  - normalise both sides: strip, lowercase, remove \\text{}, trailing '.', and
    surrounding parentheses/brackets, so "(A)", "A", "a)" and "\\boxed{(A)}" all -> "a".
  - correct iff normalised strings equal, or both parse as numbers within 1e-6.
  - no fallback to "last letter in text": a missing \\boxed{} is wrong.
Usage: python scorer_mc.py <per_task.jsonl>
"""
import json
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from scorer_math import last_boxed  # noqa: E402

SCORER_VERSION = "fresh-mc-v1"
TOL = 1e-6


def normalize(s):
    if s is None:
        return None
    s = s.strip().lower()
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = s.strip().rstrip(".").strip()
    m = re.fullmatch(r"[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?", s)
    if m and len(m.group(1)) <= 2 and not m.group(1).isdigit():
        s = m.group(1)
    elif m and m.group(1).isdigit():
        s = m.group(1)
    return s


def _float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def score_one(final_text, gold):
    pred = last_boxed(final_text)
    pn, gn = normalize(pred), normalize(gold)
    correct = False
    if pn is not None and gn is not None:
        if pn == gn:
            correct = True
        else:
            pf, gf = _float(pn), _float(gn)
            correct = pf is not None and gf is not None and abs(pf - gf) < TOL
    return {"pred": pred, "gold": gold, "correct": correct}


def score_file(path):
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("kind") != "task":
            continue
        out.append({"id": r["id"], "task": r.get("task"),
                    **score_one(r.get("final_text", ""), r["gold"])})
    return out, sum(x["correct"] for x in out), len(out)


def _selftest():
    cases = [
        ("so \\boxed{B}", "B", True),
        ("\\boxed{(A)}", "(A)", True),
        ("\\boxed{A}", "(A)", True),
        ("\\boxed{a)}", "A", True),
        ("\\boxed{Yes}", "yes", True),
        ("\\boxed{\\text{No}}", "No", True),
        ("\\boxed{True}", "False", False),
        ("\\boxed{C}", "B", False),
        ("the answer is B", "B", False),
        ("\\boxed{12.0}", "12", True),
        ("\\boxed{(A)}", "(B)", False),
    ]
    for text, gold, want in cases:
        got = score_one(text, gold)["correct"]
        assert got == want, (text, gold, got, want)
    print("scorer_mc selftest ok:", SCORER_VERSION)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _selftest()
    else:
        out, c, n = score_file(sys.argv[1])
        print(f"{SCORER_VERSION}: {c}/{n} correct")
