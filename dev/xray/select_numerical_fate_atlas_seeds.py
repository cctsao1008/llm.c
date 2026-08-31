#!/usr/bin/env python3
"""Select flip cases and matched non-flip controls for a Numerical Fate Atlas.

The selector deliberately does not infer mechanism.  It converts the existing
population landscape CSV into a small causal-trajectory manifest.

For every selected (position, perturbation-family) flip, controls are chosen
from positions that:
  * do not flip under the same family,
  * were inside the landscape near-margin set,
  * passed repeatability,
  * and, by default, did not flip under any surveyed family.

Matching is hierarchical rather than hidden in one arbitrary scalar metric:
  1. same reference top1/runner pair and within --max-t-diff,
  2. same reference top1/runner pair at any local t,
  3. any reference pair within --max-t-diff,
  4. any eligible near-margin control.
Within the first non-empty scope, nearest log10 baseline margin is primary and
nearest local t is secondary.  The manifest records the scope and distances so
matching quality remains auditable.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_FAMILIES = ["l00-qkv", "l00-attproj", "l00-fc", "l00-fcproj"]


def i(row, key):
    return int(row[key])


def f(row, key):
    return float(row[key])


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {
            "batch", "family", "scan_index", "b", "t", "input_token",
            "ref_top1", "ref_runner", "ref_top2_margin", "alt_top1",
            "alt_runner", "alt_top2_margin", "flip", "competitor",
            "ref_pair_margin", "alt_pair_margin", "pair_shift",
            "near_ref", "repeat_exact",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"missing CSV columns: {sorted(missing)}")
        return list(reader)


def short_family(name):
    return name[4:] if name.startswith("l00-") else name


def parse_scan_indexes(values):
    out = []
    for value in values or []:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def control_sort_key(flip_row, candidate):
    mf = f(flip_row, "ref_top2_margin")
    mc = f(candidate, "ref_top2_margin")
    if mf > 0.0 and mc > 0.0:
        log_dist = abs(math.log10(mc) - math.log10(mf))
    else:
        log_dist = math.inf
    return (log_dist, abs(i(candidate, "t") - i(flip_row, "t")),
            abs(i(candidate, "scan_index") - i(flip_row, "scan_index")))


def choose_controls(flip_row, candidates, count, max_t_diff):
    same_pair = lambda r: (
        i(r, "ref_top1") == i(flip_row, "ref_top1") and
        i(r, "ref_runner") == i(flip_row, "ref_runner")
    )
    near_t = lambda r: abs(i(r, "t") - i(flip_row, "t")) <= max_t_diff

    scopes = [
        ("same_pair_within_t", lambda r: same_pair(r) and near_t(r)),
        ("same_pair_any_t", same_pair),
        ("any_pair_within_t", near_t),
        ("any_pair_any_t", lambda r: True),
    ]
    for scope_name, pred in scopes:
        pool = [r for r in candidates if pred(r)]
        if pool:
            pool.sort(key=lambda r: control_sort_key(flip_row, r))
            return scope_name, pool[:count]
    return "none", []


def manifest_record(row, *, role, pair_id, B, T, multiplicity,
                    matched_to="", match_scope="", margin_log10_distance="",
                    margin_ratio="", t_distance=""):
    return {
        "pair_id": pair_id,
        "role": role,
        "family": short_family(row["family"]),
        "landscape_family": row["family"],
        "B": B,
        "T": T,
        "scan_index": i(row, "scan_index"),
        "batch": i(row, "batch"),
        "target": i(row, "b") * T + i(row, "t"),
        "b": i(row, "b"),
        "t": i(row, "t"),
        "input_token": i(row, "input_token"),
        "ref_top1": i(row, "ref_top1"),
        "ref_runner": i(row, "ref_runner"),
        "ref_top2_margin": f(row, "ref_top2_margin"),
        "landscape_flip": i(row, "flip"),
        "multiplicity": multiplicity,
        "matched_to_scan_index": matched_to,
        "match_scope": match_scope,
        "margin_log10_distance": margin_log10_distance,
        "margin_ratio": margin_ratio,
        "t_distance": t_distance,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Select flip + matched-control seeds for numerical fate trajectories")
    ap.add_argument("csv", help="CSV from numerical_fate_landscape_probe")
    ap.add_argument("--out", default="/tmp/numerical_fate_atlas_seeds.csv")
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument(
        "--scan-index", action="append", default=[],
        help="select one or more scan indexes; repeat or pass comma-separated values. "
             "If omitted, all flipped positions are selected.")
    ap.add_argument("--controls-per-flip", type=int, default=1)
    ap.add_argument("--max-t-diff", type=int, default=64)
    ap.add_argument(
        "--allow-vulnerable-controls", action="store_true",
        help="allow a control that is non-flip for this family but flips under another family")
    args = ap.parse_args()

    if args.B <= 0 or args.T <= 0 or args.controls_per_flip < 0 or args.max_t_diff < 0:
        raise SystemExit("B/T must be positive and control counts/distances non-negative")

    rows = read_rows(args.csv)
    if not rows:
        raise SystemExit("empty CSV")

    by_pos = defaultdict(dict)
    for row in rows:
        by_pos[i(row, "scan_index")][row["family"]] = row

    families = [fam for fam in DEFAULT_FAMILIES if any(r["family"] == fam for r in rows)]
    families += sorted({r["family"] for r in rows} - set(families))

    incomplete = [
        scan for scan, fam_rows in by_pos.items()
        if set(fam_rows) != set(families)
    ]
    if incomplete:
        raise SystemExit(f"{len(incomplete)} positions do not contain all expected families")

    global_flip = {
        scan for scan, fam_rows in by_pos.items()
        if any(i(r, "flip") == 1 for r in fam_rows.values())
    }
    multiplicity = {
        scan: sum(i(r, "flip") == 1 for r in fam_rows.values())
        for scan, fam_rows in by_pos.items()
    }

    requested = parse_scan_indexes(args.scan_index)
    if requested:
        missing = [scan for scan in requested if scan not in by_pos]
        if missing:
            raise SystemExit(f"requested scan indexes not found: {missing}")
        selected_positions = requested
    else:
        selected_positions = sorted(global_flip)

    manifest = []
    selected_flip_memberships = 0
    unmatched = 0

    for scan in selected_positions:
        fam_rows = by_pos[scan]
        for fam in families:
            flip_row = fam_rows[fam]
            if i(flip_row, "flip") != 1:
                continue
            selected_flip_memberships += 1
            pair_id = f"{scan}:{fam}"
            manifest.append(manifest_record(
                flip_row, role="flip", pair_id=pair_id, B=args.B, T=args.T,
                multiplicity=multiplicity[scan]))

            if args.controls_per_flip == 0:
                continue

            candidates = []
            for cscan, cfams in by_pos.items():
                crow = cfams[fam]
                if i(crow, "flip") != 0:
                    continue
                if i(crow, "near_ref") != 1 or i(crow, "repeat_exact") != 1:
                    continue
                if not args.allow_vulnerable_controls and cscan in global_flip:
                    continue
                candidates.append(crow)

            scope, controls = choose_controls(
                flip_row, candidates, args.controls_per_flip, args.max_t_diff)
            if not controls:
                unmatched += 1
                continue

            mf = f(flip_row, "ref_top2_margin")
            for ci, control in enumerate(controls, start=1):
                mc = f(control, "ref_top2_margin")
                log_dist = (
                    abs(math.log10(mc) - math.log10(mf))
                    if mf > 0.0 and mc > 0.0 else math.inf
                )
                ratio = mc / mf if mf != 0.0 else math.inf
                manifest.append(manifest_record(
                    control, role=f"control{ci}", pair_id=pair_id,
                    B=args.B, T=args.T, multiplicity=multiplicity[i(control, "scan_index")],
                    matched_to=scan, match_scope=scope,
                    margin_log10_distance=log_dist,
                    margin_ratio=ratio,
                    t_distance=abs(i(control, "t") - i(flip_row, "t"))))

    fieldnames = [
        "pair_id", "role", "family", "landscape_family", "B", "T",
        "scan_index", "batch", "target", "b", "t", "input_token",
        "ref_top1", "ref_runner", "ref_top2_margin", "landscape_flip",
        "multiplicity", "matched_to_scan_index", "match_scope",
        "margin_log10_distance", "margin_ratio", "t_distance",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    controls = sum(r["role"].startswith("control") for r in manifest)
    print(
        "[xray][fate-atlas-seeds] "
        f"input={args.csv} out={args.out} selected_positions={len(selected_positions)} "
        f"flip_memberships={selected_flip_memberships} controls={controls} "
        f"unmatched_flip_memberships={unmatched} global_clean_controls={0 if args.allow_vulnerable_controls else 1}"
    )
    for row in manifest:
        if row["role"] == "flip":
            print(
                "[xray][fate-atlas-seed] "
                f"pair_id={row['pair_id']} role=flip family={row['family']} "
                f"scan_index={row['scan_index']} batch={row['batch']} target={row['target']} "
                f"t={row['t']} margin={row['ref_top2_margin']:.9e} multiplicity={row['multiplicity']}"
            )
        else:
            print(
                "[xray][fate-atlas-seed] "
                f"pair_id={row['pair_id']} role={row['role']} family={row['family']} "
                f"scan_index={row['scan_index']} batch={row['batch']} target={row['target']} "
                f"t={row['t']} margin={row['ref_top2_margin']:.9e} "
                f"scope={row['match_scope']} log_margin_distance={float(row['margin_log10_distance']):.6f} "
                f"margin_ratio={float(row['margin_ratio']):.6f} t_distance={row['t_distance']}"
            )


if __name__ == "__main__":
    main()
