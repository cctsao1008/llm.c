#!/usr/bin/env python3
"""Run terminal readout decomposition across a Numerical Fate Atlas manifest.

Each selected manifest row is evaluated with numerical_fate_terminal_readout_probe.
The runner keeps exact per-case output in a log and writes one compact CSV row
per case.  It does not infer a Transformer mechanism; it only classifies where
switching the terminal arithmetic is sufficient to change the final decision.
"""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

SUMMARY_TAG = "[xray][terminal-readout-summary]"
VALIDITY_TAG = "[xray][terminal-readout-validity]"


def parse_kv_line(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key] = value
    return out


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def classify(summary: dict[str, str]) -> str:
    if summary.get("case_valid") != "1":
        return "invalid"
    if summary.get("classifier_switch_restores_ref") == "1":
        return "classifier-switch-restores-ref"
    if summary.get("terminal_cpu64_restores_ref") == "1":
        if summary.get("lnf_added_changes_winner") == "1":
            return "lnf-needed-to-restore-ref"
        return "cpu64-terminal-restores-ref-other"
    return "cpu64-terminal-stays-alt"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run terminal-readout audit over atlas manifest")
    ap.add_argument("manifest")
    ap.add_argument("--exe", required=True)
    ap.add_argument("--log", default="/tmp/numerical_fate_terminal_atlas.log")
    ap.add_argument("--summary-csv", default="/tmp/numerical_fate_terminal_atlas.csv")
    ap.add_argument("--roles", default="flip",
                    help="comma-separated manifest roles; default: flip")
    ap.add_argument("--max-cases", type=int, default=0,
                    help="0 means all selected rows")
    args = ap.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("empty manifest")

    wanted_roles = {x.strip() for x in args.roles.split(",") if x.strip()}
    if wanted_roles:
        rows = [r for r in rows if r.get("role", "") in wanted_roles]
    if args.max_cases > 0:
        rows = rows[: args.max_cases]
    if not rows:
        raise SystemExit("no manifest rows selected")

    required = {"pair_id", "role", "family", "scan_index", "B", "T", "batch", "target"}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"manifest missing columns: {sorted(missing)}")

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, str]] = []
    failures = 0
    invalid = 0

    with open(args.log, "w", encoding="utf-8") as log:
        for case_no, row in enumerate(rows, start=1):
            cmd = [
                args.exe,
                row["B"], row["T"], row["batch"], row["target"], row["family"],
            ]
            header = (
                f"[xray][terminal-atlas-case] case={case_no}/{len(rows)} "
                f"pair_id={row['pair_id']} role={row['role']} family={row['family']} "
                f"scan_index={row['scan_index']} command={shlex.join(cmd)}"
            )
            print(header)
            log.write(header + "\n")
            log.flush()

            proc = subprocess.run(cmd, text=True, capture_output=True)
            log.write(proc.stdout)
            if proc.stdout and not proc.stdout.endswith("\n"):
                log.write("\n")
            if proc.stderr:
                log.write("[stderr]\n" + proc.stderr)
                if not proc.stderr.endswith("\n"):
                    log.write("\n")
            log.write(f"[xray][terminal-atlas-case-end] returncode={proc.returncode}\n\n")
            log.flush()

            if proc.returncode != 0:
                failures += 1
                print(f"[xray][terminal-atlas-case] FAILED returncode={proc.returncode}")
                continue

            summary_line = None
            validity_line = None
            for line in proc.stdout.splitlines():
                if line.startswith(VALIDITY_TAG):
                    validity_line = line
                if line.startswith(SUMMARY_TAG) and "gpu_pair=" in line:
                    summary_line = line
            if summary_line is None or validity_line is None:
                failures += 1
                print("[xray][terminal-atlas-case] FAILED missing summary/validity line")
                continue

            s = parse_kv_line(summary_line)
            v = parse_kv_line(validity_line)
            regime = classify(s)
            if s.get("case_valid") != "1":
                invalid += 1

            gpu_pair = as_float(s.get("gpu_pair", ""))
            cpu_cls_pair = as_float(s.get("cpu64_classifier_pair", ""))
            cpu_terminal_pair = as_float(s.get("cpu64_terminal_pair", ""))
            cls_shift = cpu_cls_pair - gpu_pair
            lnf_shift = cpu_terminal_pair - cpu_cls_pair
            shift_over_gpu_margin = (
                abs(cls_shift) / abs(gpu_pair)
                if math.isfinite(gpu_pair) and gpu_pair != 0.0 and math.isfinite(cls_shift)
                else math.nan
            )

            rec = {
                "pair_id": row["pair_id"],
                "role": row["role"],
                "family": row["family"],
                "scan_index": row["scan_index"],
                "batch": row["batch"],
                "target": row["target"],
                "ref": s.get("ref", ""),
                "alt": s.get("alt", ""),
                "gpu_pair": s.get("gpu_pair", ""),
                "cpu64_classifier_pair": s.get("cpu64_classifier_pair", ""),
                "cpu64_terminal_pair": s.get("cpu64_terminal_pair", ""),
                "classifier_pair_shift": f"{cls_shift:.17g}" if math.isfinite(cls_shift) else "",
                "lnf_added_pair_shift": f"{lnf_shift:.17g}" if math.isfinite(lnf_shift) else "",
                "abs_classifier_shift_over_abs_gpu_pair": (
                    f"{shift_over_gpu_margin:.17g}" if math.isfinite(shift_over_gpu_margin) else ""
                ),
                "classifier_switch_restores_ref": s.get("classifier_switch_restores_ref", ""),
                "terminal_cpu64_restores_ref": s.get("terminal_cpu64_restores_ref", ""),
                "lnf_added_changes_winner": s.get("lnf_added_changes_winner", ""),
                "terminal_regime": regime,
                "ref_repeat_exact": v.get("ref_repeat_exact", ""),
                "alt_repeat_exact": v.get("alt_repeat_exact", ""),
                "l11_gpu_replay_exact": v.get("l11_gpu_replay_exact", ""),
                "case_valid": s.get("case_valid", ""),
            }
            summaries.append(rec)
            print(
                "[xray][terminal-atlas-result] "
                f"scan_index={row['scan_index']} family={row['family']} "
                f"regime={regime} gpu_pair={s.get('gpu_pair','?')} "
                f"cpu64_classifier_pair={s.get('cpu64_classifier_pair','?')} "
                f"cpu64_terminal_pair={s.get('cpu64_terminal_pair','?')} "
                f"case_valid={s.get('case_valid','?')}"
            )

    fieldnames = list(summaries[0].keys()) if summaries else [
        "pair_id", "role", "family", "scan_index", "terminal_regime", "case_valid"
    ]
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    counts = Counter(r["terminal_regime"] for r in summaries)
    scans_by_regime: dict[str, set[str]] = defaultdict(set)
    for r in summaries:
        scans_by_regime[r["terminal_regime"]].add(r["scan_index"])

    print(
        "[xray][terminal-atlas-summary] "
        f"requested={len(rows)} completed={len(summaries)} failures={failures} invalid={invalid} "
        f"unique_scan_positions={len({r['scan_index'] for r in summaries})} "
        f"log={args.log} summary_csv={args.summary_csv}"
    )
    for regime, n in counts.most_common():
        print(
            "[xray][terminal-atlas-regime] "
            f"regime={regime} memberships={n} unique_scan_positions={len(scans_by_regime[regime])}"
        )

    if failures or invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
