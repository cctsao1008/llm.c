#!/usr/bin/env python3
"""Classify Numerical Fate Atlas trajectories into descriptive dynamical regimes.

This analysis deliberately separates trajectory memberships from unique scan
positions. Multiple perturbation-family memberships at one scan_index share the
same underlying token/context and therefore are not independent observations.
No inferential statistic in this script treats those memberships as iid.

The regime labels describe the CPU64-suffix pair topology of an exact natural
GPU trajectory; they are not operator or mechanism labels.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def as_int(row, key, default=0):
    value = row.get(key, "")
    return int(value) if value not in (None, "") else default


def source_branch(family):
    if family in {"qkv", "attproj"}:
        return "attention-branch"
    if family in {"fc", "fcproj"}:
        return "mlp-branch"
    return "other"


def classify(row):
    topo = row.get("pair_topology", "")
    sign_changes = as_int(row, "pair_sign_changes", -1)
    final_flip = as_int(row, "final_flip", 0)

    if not topo:
        return "invalid-empty-topology"

    # A natural GPU execution flipped, yet none of the exact residual3
    # checkpoints evaluated with the validated CPU64 suffix ever places the
    # selected reference-vs-GPU-winner pair on the alternate side.
    if final_flip and "-" not in topo and "0" not in topo:
        return "cpu64-never-alt"

    if set(topo) == {"-"}:
        return "state-persistent-from-l0"

    # Strict one-way acquisition: one positive prefix, then one negative tail.
    if sign_changes == 1 and topo[0] == "+" and topo[-1] == "-" and "0" not in topo:
        first_neg = topo.find("-")
        if first_neg > 0 and set(topo[:first_neg]) == {"+"} and set(topo[first_neg:]) == {"-"}:
            return "monotone-acquisition"

    if sign_changes >= 2:
        return "reversible-multicrossing"

    if "0" in topo:
        return "boundary-contact-other"

    return "other"


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {
            "role", "family", "scan_index", "final_flip", "pair_topology",
            "winner_topology", "pair_sign_changes", "case_valid",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"missing CSV columns: {sorted(missing)}")
        return list(reader)


def main():
    ap = argparse.ArgumentParser(description="Classify Numerical Fate Atlas regimes")
    ap.add_argument("csv", help="summary CSV from run_numerical_fate_atlas_manifest.py")
    ap.add_argument("--report", default=None, help="optional Markdown report")
    ap.add_argument("--classified-csv", default=None,
                    help="optional copy of input rows with source_branch and regime columns")
    args = ap.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        raise SystemExit("empty atlas summary")

    invalid = [r for r in rows if as_int(r, "case_valid") != 1]
    flips = [r for r in rows if r["role"] == "flip"]
    valid_flips = [r for r in flips if as_int(r, "case_valid") == 1]

    for r in valid_flips:
        r["source_branch"] = source_branch(r["family"])
        r["regime"] = classify(r)

    by_scan = defaultdict(list)
    for r in valid_flips:
        by_scan[as_int(r, "scan_index")].append(r)

    regime_order = [
        "cpu64-never-alt",
        "state-persistent-from-l0",
        "monotone-acquisition",
        "reversible-multicrossing",
        "boundary-contact-other",
        "other",
        "invalid-empty-topology",
    ]
    membership_counts = Counter(r["regime"] for r in valid_flips)
    scan_presence = Counter()
    for group in by_scan.values():
        for regime in {r["regime"] for r in group}:
            scan_presence[regime] += 1

    lines = []
    emit = lines.append
    emit("# Numerical Fate Regime Analysis")
    emit("")
    emit(f"Input: `{args.csv}`")
    emit("")
    emit("## Validity and independence boundary")
    emit("")
    emit(f"- Input trajectory rows: {len(rows)}")
    emit(f"- Flip memberships: {len(flips)}")
    emit(f"- Valid flip memberships: {len(valid_flips)}")
    emit(f"- Unique flipped scan positions: {len(by_scan)}")
    emit(f"- Invalid trajectories: {len(invalid)}")
    emit("")
    emit("**Independence warning:** perturbation-family memberships sharing one `scan_index` are clustered observations of the same token/context. Membership counts below are descriptive only; they must not be treated as iid samples for a significance test.")
    emit("")

    emit("## Descriptive regime definitions")
    emit("")
    emit("- `cpu64-never-alt`: the natural GPU execution flips, but no exact residual3 checkpoint evaluated through the CPU64 suffix reaches or crosses the selected reference-vs-GPU-winner pair boundary.")
    emit("- `state-persistent-from-l0`: every CPU64 suffix checkpoint is already on the alternate side (`------------` for a 12-layer model).")
    emit("- `monotone-acquisition`: one positive prefix followed by one permanently negative suffix.")
    emit("- `reversible-multicrossing`: the pair sign changes at least twice; fate is lost, recovered, or reacquired across depth.")
    emit("- `boundary-contact-other` / `other`: valid trajectories not captured by the three coarse directional patterns above.")
    emit("")

    emit("## Regime counts")
    emit("")
    emit("| regime | trajectory memberships | unique scan positions containing regime |")
    emit("|---|---:|---:|")
    for regime in regime_order:
        if membership_counts[regime] or scan_presence[regime]:
            emit(f"| {regime} | {membership_counts[regime]} | {scan_presence[regime]} |")
    emit("")

    emit("## Regime by perturbation family")
    emit("")
    families = sorted({r["family"] for r in valid_flips})
    observed_regimes = [r for r in regime_order if membership_counts[r]]
    emit("| family | n | " + " | ".join(observed_regimes) + " |")
    emit("|---|---:|" + "---:|" * len(observed_regimes))
    for fam in families:
        group = [r for r in valid_flips if r["family"] == fam]
        c = Counter(r["regime"] for r in group)
        emit("| " + fam + f" | {len(group)} | " + " | ".join(str(c[x]) for x in observed_regimes) + " |")
    emit("")

    emit("## Unique-position regime composition")
    emit("")
    emit("| scan_index | memberships | families | regimes | homogeneous regime |")
    emit("|---:|---:|---|---|---:|")
    heterogeneous = 0
    for scan in sorted(by_scan):
        group = by_scan[scan]
        regimes = sorted({r["regime"] for r in group})
        homogeneous = len(regimes) == 1
        heterogeneous += int(not homogeneous)
        fams = ";".join(r["family"] for r in group)
        detail = ";".join(f"{r['family']}:{r['regime']}" for r in group)
        emit(f"| {scan} | {len(group)} | {fams} | {detail} | {int(homogeneous)} |")
    emit("")
    emit(f"- Multi-membership/unique positions with heterogeneous regimes: {heterogeneous}")
    emit("")

    emit("## Cross-branch within-position contrasts")
    emit("")
    emit("Only positions containing at least one attention-branch and one MLP-branch flip are shown. This is a within-token descriptive contrast and still does not establish a source-branch mechanism.")
    emit("")
    emit("| scan_index | attention-branch | mlp-branch | same regime set |")
    emit("|---:|---|---|---:|")
    cross_n = 0
    cross_same = 0
    for scan in sorted(by_scan):
        group = by_scan[scan]
        att = [r for r in group if r["source_branch"] == "attention-branch"]
        mlp = [r for r in group if r["source_branch"] == "mlp-branch"]
        if not att or not mlp:
            continue
        cross_n += 1
        aset = {r["regime"] for r in att}
        mset = {r["regime"] for r in mlp}
        same = aset == mset
        cross_same += int(same)
        adetail = ";".join(f"{r['family']}:{r['regime']}" for r in att)
        mdetail = ";".join(f"{r['family']}:{r['regime']}" for r in mlp)
        emit(f"| {scan} | {adetail} | {mdetail} | {int(same)} |")
    emit("")
    emit(f"- Cross-branch positions: {cross_n}")
    emit(f"- Cross-branch positions with identical regime sets: {cross_same}")
    emit("")

    emit("## Interpretation gate")
    emit("")
    emit("These regimes classify direct CPU64-suffix fate trajectories from exact executed checkpoints. They do not identify a causal operator. In particular, a `cpu64-never-alt` case says that the final GPU decision is not represented as an alternate pair decision under the CPU64 suffix at any sampled residual3 checkpoint; it motivates finer terminal-suffix execution analysis rather than a claim that the classifier alone is responsible.")

    text = "\n".join(lines) + "\n"
    print(text, end="")

    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
        print(f"[xray][fate-regime-analysis] report={args.report}")

    if args.classified_csv:
        if not valid_flips:
            raise SystemExit("no valid flip rows to write")
        fieldnames = list(valid_flips[0].keys())
        with open(args.classified_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(valid_flips)
        print(f"[xray][fate-regime-analysis] classified_csv={args.classified_csv}")


if __name__ == "__main__":
    main()
