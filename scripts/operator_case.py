"""Search and validate the evidence-backed Phase 3 operator case library."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "operator_cases"

def cases():
    for path in sorted(CASE_DIR.glob("*.json")):
        yield path, json.loads(path.read_text())

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    s = sub.add_parser("search")
    s.add_argument("--operator")
    s.add_argument("--mode")
    s.add_argument("--device")
    s.add_argument("--status")
    sub.add_parser("verify")
    a = p.parse_args()
    if a.command == "verify":
        errors = []
        for path, case in cases():
            for evidence in case.get("evidence", []):
                if not (ROOT / evidence).exists():
                    errors.append(f"{path.name}: missing evidence {evidence}")
            for key in ("name", "operator", "mode", "device", "dtype", "status", "contract", "gates", "lesson"):
                if key not in case:
                    errors.append(f"{path.name}: missing field {key}")
        if errors:
            print("\n".join(errors))
            raise SystemExit(1)
        print(f"verified {sum(1 for _ in cases())} cases")
        return
    selected = list(cases())
    if a.command == "search":
        selected = [(p, c) for p, c in selected
                    if (not a.operator or a.operator in c["operator"])
                    and (not a.mode or a.mode == c["mode"])
                    and (not a.device or a.device == c["device"])
                    and (not a.status or a.status == c["status"])]
    for path, c in selected:
        print(json.dumps({"file": str(path.relative_to(ROOT)), "name": c["name"],
                          "operator": c["operator"], "mode": c["mode"],
                          "status": c["status"]}, ensure_ascii=False))

if __name__ == "__main__":
    main()
