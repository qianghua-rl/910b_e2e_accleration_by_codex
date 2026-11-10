"""Offline scorer for MATH-500 style answers (LaTeX, not only numbers).

Version: fresh-math-v1. Separate from scorer.py (fresh-v1, numeric-only, frozen).
Re-scores a per_task.jsonl produced by cb_run.py; the inline `correct` field
written by cb_run.py is NOT used for these benchmarks.

Rules:
  - prediction = content of the LAST \\boxed{...} in final_text, with balanced
    braces (fresh-v1's regex cannot handle \\frac{a}{b}).
  - both sides normalised (Hendrycks-MATH style): drop spaces, \\left/\\right,
    \\dfrac/\\tfrac -> \\frac, \\$, ^\\circ, ^{\\circ}, \\text{...} units, trailing '.',
    leading 'x=', thousands commas inside numbers, \\frac{a}{b} -> a/b, \\sqrt2 -> \\sqrt{2}.
  - correct iff normalised strings are equal, OR both parse to floats within 1e-6
    (after evaluating a/b).
Usage: python scorer_math.py <per_task.jsonl>
"""
import json
import re
import sys

SCORER_VERSION = "fresh-math-v2"
TOL = 1e-6


def _boxed_at(text, idx):
    """Content of the \\boxed{...} starting at idx (balanced braces), or None if unterminated."""
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j].strip()
    return None


def last_boxed(text):
    """Content of the last COMPLETE \\boxed{...} (balanced braces), or None.
    v2: a dangling/unterminated \\boxed (e.g. output cut mid-repetition) is skipped
    instead of making the whole answer None."""
    if not text:
        return None
    idx = text.rfind("\\boxed")
    while idx >= 0:
        val = _boxed_at(text, idx)
        if val is not None:
            return val
        idx = text.rfind("\\boxed", 0, idx)
    return None


def _frac_to_slash(s):
    # \frac{a}{b} -> (a)/(b), repeatedly for nested simple cases
    pat = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
    prev = None
    while prev != s:
        prev = s
        s = pat.sub(r"(\1)/(\2)", s)
    return s


def normalize(s):
    if s is None:
        return None
    s = s.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"\^\{?\\circ\}?", "", s)
    s = re.sub(r"\\text\{\s*[a-zA-Z ]*\}", "", s)   # trailing units like \text{ cm}
    s = re.sub(r"\\mbox\{[^{}]*\}", "", s)
    s = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", s)
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    s = re.sub(r"^[a-zA-Z]=", "", s)
    s = _frac_to_slash(s)
    # thousands separators like 1,000 or 1{,}000 (only when they look like groups)
    s = s.replace("{,}", ",")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    return s


def _to_float(s):
    if s is None:
        return None
    m = re.fullmatch(r"\(?(-?[\d.]+)\)?/\(?(-?[\d.]+)\)?", s)
    try:
        if m:
            return float(m.group(1)) / float(m.group(2))
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def score_one(final_text, gold):
    pred = last_boxed(final_text)
    pn, gn = normalize(pred), normalize(gold)
    correct = False
    if pn is not None and gn is not None:
        if pn == gn:
            correct = True
        else:
            pf, gf = _to_float(pn), _to_float(gn)
            correct = pf is not None and gf is not None and abs(pf - gf) < TOL
    return {"pred": pred, "gold": gold, "correct": correct}


def score_file(path):
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("kind") != "task" and "final_text" not in r:
            continue
        out.append({"id": r["id"], **score_one(r.get("final_text", ""), r["gold"])})
    return out, sum(x["correct"] for x in out), len(out)


def _selftest():
    cases = [
        ("so \\boxed{\\frac{14}{3}}", "\\frac{14}{3}", True),
        ("\\boxed{\\dfrac{14}{3}}", "\\frac{14}{3}", True),
        ("\\boxed{14/3}", "\\frac{14}{3}", True),
        ("\\boxed{90^\\circ}", "90^\\circ", True),
        ("\\boxed{90}", "90^\\circ", True),
        ("\\boxed{(3, \\frac{\\pi}{2})}", "\\left( 3, \\frac{\\pi}{2} \\right)", True),
        ("\\boxed{1,000}", "1000", True),
        ("\\boxed{x=5}", "5", True),
        ("\\boxed{3\\sqrt{13}}", "3\\sqrt{13}", True),
        ("\\boxed{\\sqrt2}", "\\sqrt{2}", True),
        ("\\boxed{7}", "8", False),
        ("the answer is 5", "5", False),
        ("\\boxed{}", "8", False),
        ("first \\boxed{3} then \\boxed{\\frac{1}{2}}", "3", False),
    ]
    for text, gold, want in cases:
        got = score_one(text, gold)["correct"]
        assert got == want, (text, gold, got, want)
    print("scorer_math selftest ok:", SCORER_VERSION)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _selftest()
    else:
        out, c, n = score_file(sys.argv[1])
        print(f"{SCORER_VERSION}: {c}/{n} correct")
        for x in out:
            if not x["correct"]:
                print(" wrong:", x)
