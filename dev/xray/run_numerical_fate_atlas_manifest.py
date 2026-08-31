#!/usr/bin/env python3
"""Run numerical_fate_trajectory_probe over a seed manifest and collect topology.

This runner compiles nothing; the shell wrapper builds the CUDA probe once and
passes its executable here.  Full per-layer output is preserved in one log,
while a compact CSV collects one topology summary per manifest row.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from pathlib import Path


TAG = "[xray][fate-trajectory-summary]"


def parse_kv_line(line):
    out = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key] = value
    return out


def main():
    ap = argparse.ArgumentParser(description="Run numerical fate trajectory manifest")
    ap.add_argument("manifest")
    ap.add_argument("--exe", required=True)
    ap.add_argument("--log", default="/tmp/numerical_fate_atlas_trajectory.log")
    ap.add_argument("--summary-csv", default="/tmp/numerical_fate_atlas_trajectory_summary.csv")
    ap.add_argument("--max-cases", type=int, default=0,
                    help="0 means all manifest rows")
    ap.add_argument("--roles", default="",
                    help="optional comma-separated role filter, e.g. flip or flip,control1")
    args = ap.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("empty manifest")

    wanted_roles = {x.strip() for x in args.roles.split(",") if x.strip()}
    if wanted_roles:
        rows = [r for r in rows if r["role"] in wanted_roles]
    if args.max_cases > 0:
        rows = rows[:args.max_cases]
    if not rows:
        raise SystemExit("no manifest rows selected")

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    run_failures = 0
    invalid_cases = 0

    with open(args.log, "w", encoding="utf-8") as log:
        for case_no, row in enumerate(rows, start=1):
            cmd = [
                args.exe,
                row["B"], row["T"], row["batch"], row["target"], row["family"],
            ]
            header = (
                f"[xray][fate-atlas-run-case] case={case_no}/{len(rows)} "
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
                log.write("[stderr]\n")
                log.write(proc.stderr)
                if not proc.stderr.endswith("\n"):
                    log.write("\n")
            log.write(f"[xray][fate-atlas-run-case-end] returncode={proc.returncode}\n\n")
            log.flush()

            if proc.returncode != 0:
                run_failures += 1
                print(f"[xray][fate-atlas-run-case] FAILED returncode={proc.returncode}")
                continue

            summary_line = None
            for line in proc.stdout.splitlines():
                if line.startswith(TAG) and "pair_topology=" in line:
                    summary_line = line
            if summary_line is None:
                run_failures += 1
                print("[xray][fate-atlas-run-case] FAILED missing topology summary")
                continue

            parsed = parse_kv_line(summary_line)
            case_valid = parsed.get("case_valid") == "1"
            if not case_valid:
                invalid_cases += 1

            summaries.append({
                "pair_id": row["pair_id"],
                "role": row["role"],
                "family": row["family"],
                "scan_index": row["scan_index"],
                "batch": row["batch"],
                "target": row["target"],
                "b": row["b"],
                "t": row["t"],
                "ref_top2_margin": row["ref_top2_margin"],
                "multiplicity": row["multiplicity"],
                "matched_to_scan_index": row["matched_to_scan_index"],
                "match_scope": row["match_scope"],
                "margin_log10_distance": row["margin_log10_distance"],
                "margin_ratio": row["margin_ratio"],
                "t_distance": row["t_distance"],
                "final_flip": parsed.get("final_flip", ""),
                "ref": parsed.get("ref", ""),
                "competitor": parsed.get("competitor", ""),
                "competitor_source": parsed.get("competitor_source", ""),
                "pair_topology": parsed.get("pair_topology", ""),
                "winner_topology": parsed.get("winner_topology", ""),
                "winner_changes": parsed.get("winner_changes", ""),
                "pair_sign_changes": parsed.get("pair_sign_changes", ""),
                "last_winner_change_layer": parsed.get("last_winner_change_layer", ""),
                "last_pair_sign_change_layer": parsed.get("last_pair_sign_change_layer", ""),
                "stable_ref_from_layer": parsed.get("stable_ref_from_layer", ""),
                "stable_competitor_from_layer": parsed.get("stable_competitor_from_layer", ""),
                "stable_pair_positive_from_layer": parsed.get("stable_pair_positive_from_layer", ""),
                "stable_pair_zero_from_layer": parsed.get("stable_pair_zero_from_layer", ""),
                "stable_pair_negative_from_layer": parsed.get("stable_pair_negative_from_layer", ""),
                "min_abs_pair": parsed.get("min_abs_pair", ""),
                "min_abs_pair_layer": parsed.get("min_abs_pair_layer", ""),
                "max_abs_pair_step": parsed.get("max_abs_pair_step", ""),
                "max_abs_pair_step_layer": parsed.get("max_abs_pair_step_layer", ""),
                "final_cpu64_top1": parsed.get("final_cpu64_top1", ""),
                "final_cpu64_pair": parsed.get("final_cpu64_pair", ""),
                "repeat_valid": parsed.get("repeat_valid", ""),
                "replay_exact": parsed.get("replay_exact", ""),
                "case_valid": parsed.get("case_valid", ""),
                "elapsed_ms": parsed.get("elapsed_ms", ""),
            })
            print(
                "[xray][fate-atlas-run-result] "
                f"pair_id={row['pair_id']} role={row['role']} family={row['family']} "
                f"topology={parsed.get('pair_topology', '?')} "
                f"winner_topology={parsed.get('winner_topology', '?')} "
                f"sign_changes={parsed.get('pair_sign_changes', '?')} "
                f"case_valid={parsed.get('case_valid', '?')}"
            )

    fieldnames = list(summaries[0].keys()) if summaries else [
        "pair_id", "role", "family", "scan_index", "case_valid"
    ]
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(
        "[xray][fate-atlas-run-summary] "
        f"requested={len(rows)} completed={len(summaries)} "
        f"run_failures={run_failures} invalid_cases={invalid_cases} "
        f"log={args.log} summary_csv={args.summary_csv}"
    )
    if run_failures or invalid_cases:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
