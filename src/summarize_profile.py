"""Attribute device kernel time to labelled host ranges in a Chrome trace.

Labels come from profile_decode.py: host events named "linear_attn:...", "self_attn:...",
"mlp:...". Device kernels are attributed to a label when their start time falls inside
that label's host range. Kernels outside any label go to "other".

Caveat: NPU execution is asynchronous, so a kernel may start after its host range has
closed. The "other" bucket can therefore be inflated. Report it, do not hide it.
Usage: python summarize_profile.py <trace.json>
"""
import bisect
import collections
import json
import sys

LABEL_PREFIXES = ("linear_attn:", "self_attn:", "mlp:")


def load_events(path):
    with open(path) as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def classify_processes(events):
    """Return the set of pids whose process_name looks like a device (NPU/Ascend)."""
    device = set()
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pname = str(e.get("args", {}).get("name", "")).lower()
            if "npu" in pname or "ascend" in pname or "device" in pname:
                device.add(e.get("pid"))
    return device


def summarize(events):
    device_pids = classify_processes(events)
    host_ranges = []   # (start, end, label) on CPU side
    kernels = []       # (start, dur, name) on device side
    for e in events:
        if e.get("ph") != "X":
            continue
        ts, dur = float(e.get("ts", 0)), float(e.get("dur", 0))
        name = e.get("name", "")
        if e.get("pid") in device_pids:
            kernels.append((ts, dur, name))
        elif name.startswith(LABEL_PREFIXES):
            host_ranges.append((ts, ts + dur, name.split(":")[0]))

    host_ranges.sort()
    starts = [r[0] for r in host_ranges]

    def label_of(t):
        # Nested labels do not occur for the chosen modules; take the latest range containing t.
        i = bisect.bisect_right(starts, t) - 1
        while i >= 0:
            s, e, lab = host_ranges[i]
            if s <= t < e:
                return lab
            if i > 0 and host_ranges[i - 1][1] < t and s < t - 1e9:
                break
            i -= 1
        return "other"

    per_label = collections.defaultdict(float)
    per_kernel = collections.defaultdict(float)
    total = 0.0
    for ts, dur, name in kernels:
        lab = label_of(ts)
        per_label[lab] += dur
        per_kernel[(lab, name)] += dur
        total += dur
    top = sorted(per_kernel.items(), key=lambda kv: -kv[1])[:15]
    return {
        "device_kernel_us_total": total,
        "by_label_us": dict(sorted(per_label.items(), key=lambda kv: -kv[1])),
        "by_label_pct": {k: (100.0 * v / total if total else 0.0) for k, v in per_label.items()},
        "top_kernels": [{"label": k[0], "kernel": k[1], "us": v} for k, v in top],
        "n_device_kernels": len(kernels),
        "n_label_ranges": len(host_ranges),
    }


def _selftest():
    ev = [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "Host"}},
        {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "NPU Device 1"}},
        {"ph": "X", "pid": 1, "name": "linear_attn:model.layers.0.linear_attn", "ts": 0, "dur": 100},
        {"ph": "X", "pid": 1, "name": "mlp:model.layers.0.mlp", "ts": 100, "dur": 100},
        {"ph": "X", "pid": 2, "name": "GatedMul", "ts": 10, "dur": 30},
        {"ph": "X", "pid": 2, "name": "MatMulV3", "ts": 120, "dur": 60},
        {"ph": "X", "pid": 2, "name": "Copy", "ts": 300, "dur": 5},
    ]
    s = summarize(ev)
    assert s["by_label_us"] == {"linear_attn": 30.0, "mlp": 60.0, "other": 5.0}, s["by_label_us"]
    assert abs(s["device_kernel_us_total"] - 95.0) < 1e-9
    print("summarize selftest ok")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _selftest()
    else:
        print(json.dumps(summarize(load_events(sys.argv[1])), indent=2))
