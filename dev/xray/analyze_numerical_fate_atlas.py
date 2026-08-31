#!/usr/bin/env python3
"""Analyze Numerical Fate Atlas trajectory summaries.

The analysis stays at the topology/decision-geometry level.  It compares final
flip trajectories with matched non-flip controls and deliberately does not turn
large depth steps or repeated topology classes into operator-level mechanisms.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


def as_int(row, key, default=0):
    value = row.get(key, "")
    return int(value) if value not in (None, "") else default


def as_float(row, key):
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(x):
    if isinstance(x, int):
        return str(x)
    if x is None or not math.isfinite(float(x)):
        return "nan"
    return f"{float(x):.9e}"


def median(values):
    x = sorted(v for v in values if math.isfinite(v))
    if not x:
        return math.nan
    n = len(x)
    return x[n // 2] if n & 1 else 0.5 * (x[n // 2 - 1] + x[n // 2])


def source_branch(family):
    if family in {"qkv", "attproj"}:
        return "attention-branch"
    if family in {"fc", "fcproj"}:
        return "mlp-branch"
    return "other"


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {
            "pair_id", "role", "family", "scan_index", "ref_top2_margin",
            "final_flip", "pair_topology", "winner_topology",
            "pair_sign_changes", "stable_ref_from_layer",
            "stable_competitor_from_layer", "min_abs_pair",
            "min_abs_pair_layer", "case_valid",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"missing CSV columns: {sorted(missing)}")
        return list(reader)


def control_reaches_boundary(row):
    topo = row.get("pair_topology", "")
    return "-" in topo or "0" in topo


def main():
    ap = argparse.ArgumentParser(description="Analyze numerical fate atlas trajectory summary")
    ap.add_argument("csv")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        raise SystemExit("empty atlas summary")

    invalid = [r for r in rows if as_int(r, "case_valid") != 1]
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r["pair_id"]].append(r)

    flips = [r for r in rows if r["role"] == "flip"]
    controls = [r for r in rows if r["role"].startswith("control")]

    lines = []
    emit = lines.append
    emit("# Numerical Fate Atlas Analysis")
    emit("")
    emit(f"Input: `{args.csv}`")
    emit("")
    emit("## Validity")
    emit("")
    emit(f"- Trajectory rows: {len(rows)}")
    emit(f"- Flip rows: {len(flips)}")
    emit(f"- Control rows: {len(controls)}")
    emit(f"- Invalid trajectories (`case_valid != 1`): {len(invalid)}")
    emit(f"- Analysis validity: {1 if not invalid else 0}")
    emit("")

    emit("## Pair-topology classes")
    emit("")
    emit("| group | topology | n |")
    emit("|---|---|---:|")
    for group_name, group_rows in [("flip", flips), ("control", controls)]:
        counts = Counter(r["pair_topology"] for r in group_rows)
        for topology, n in counts.most_common():
            emit(f"| {group_name} | `{topology}` | {n} |")
    emit("")

    emit("## Winner-topology classes")
    emit("")
    emit("`R` = reference winner, `C` = selected competitor, `O` = another token.")
    emit("")
    emit("| group | topology | n |")
    emit("|---|---|---:|")
    for group_name, group_rows in [("flip", flips), ("control", controls)]:
        counts = Counter(r["winner_topology"] for r in group_rows)
        for topology, n in counts.most_common():
            emit(f"| {group_name} | `{topology}` | {n} |")
    emit("")

    emit("## Flip topology by perturbation family")
    emit("")
    emit("| family | n | topology classes | median sign changes | median stable competitor layer | median min |pair| |")
    emit("|---|---:|---|---:|---:|---:|")
    families = sorted({r["family"] for r in rows})
    for fam in families:
        fr = [r for r in flips if r["family"] == fam]
        counts = Counter(r["pair_topology"] for r in fr)
        topo_text = "; ".join(f"`{k}`×{v}" for k, v in counts.most_common()) or "-"
        emit(
            f"| {fam} | {len(fr)} | {topo_text} | "
            f"{fmt(median([as_float(r, 'pair_sign_changes') for r in fr]))} | "
            f"{fmt(median([as_float(r, 'stable_competitor_from_layer') for r in fr]))} | "
            f"{fmt(median([as_float(r, 'min_abs_pair') for r in fr]))} |"
        )
    emit("")

    emit("## Flip topology by coarse source branch")
    emit("")
    emit("This is a descriptive grouping of the L00 injection site only: `qkv`/`attproj` are grouped as the attention branch and `fc`/`fcproj` as the MLP branch. It is not a mechanism label.")
    emit("")
    emit("| source branch | n | topology classes | median sign changes | median stable competitor layer | median min |pair| |")
    emit("|---|---:|---|---:|---:|---:|")
    branches = ["attention-branch", "mlp-branch", "other"]
    for branch in branches:
        br = [r for r in flips if source_branch(r["family"]) == branch]
        if not br:
            continue
        counts = Counter(r["pair_topology"] for r in br)
        topo_text = "; ".join(f"`{k}`×{v}" for k, v in counts.most_common()) or "-"
        emit(
            f"| {branch} | {len(br)} | {topo_text} | "
            f"{fmt(median([as_float(r, 'pair_sign_changes') for r in br]))} | "
            f"{fmt(median([as_float(r, 'stable_competitor_from_layer') for r in br]))} | "
            f"{fmt(median([as_float(r, 'min_abs_pair') for r in br]))} |"
        )
    emit("")

    emit("### Source-branch × topology counts")
    emit("")
    topologies = [t for t, _ in Counter(r["pair_topology"] for r in flips).most_common()]
    if topologies:
        emit("| source branch | " + " | ".join(f"`{t}`" for t in topologies) + " |")
        emit("|---|" + "---:|" * len(topologies))
        for branch in branches:
            br = [r for r in flips if source_branch(r["family"]) == branch]
            if not br:
                continue
            counts = Counter(r["pair_topology"] for r in br)
            emit("| " + branch + " | " + " | ".join(str(counts[t]) for t in topologies) + " |")
        emit("")

    emit("## Matched flip/control comparison")
    emit("")
    emit("A control boundary contact means its CPU64 pair topology reaches `0` or `-` at an intermediate checkpoint even though the perturbed GPU execution did not finally flip.")
    emit("")
    emit("| pair_id | family | flip scan | control scan | flip margin | control margin | flip topology | control topology | same topology | control boundary contact |")
    emit("|---|---|---:|---:|---:|---:|---|---|---:|---:|")

    matched_pairs = 0
    same_topology = 0
    control_contacts = 0
    for pair_id in sorted(by_pair):
        group = by_pair[pair_id]
        frows = [r for r in group if r["role"] == "flip"]
        crows = [r for r in group if r["role"].startswith("control")]
        if len(frows) != 1:
            continue
        flip = frows[0]
        for control in crows:
            matched_pairs += 1
            same = flip["pair_topology"] == control["pair_topology"]
            contact = control_reaches_boundary(control)
            same_topology += int(same)
            control_contacts += int(contact)
            emit(
                f"| {pair_id} | {flip['family']} | {flip['scan_index']} | {control['scan_index']} | "
                f"{fmt(as_float(flip, 'ref_top2_margin'))} | {fmt(as_float(control, 'ref_top2_margin'))} | "
                f"`{flip['pair_topology']}` | `{control['pair_topology']}` | {int(same)} | {int(contact)} |"
            )
    emit("")
    emit(f"- Matched comparisons: {matched_pairs}")
    emit(f"- Exact flip/control topology matches: {same_topology}")
    emit(f"- Controls that transiently reach/contact the selected pair boundary: {control_contacts}")
    emit("")

    emit("## Margin and trajectory diagnostics")
    emit("")
    emit("| group | n | median baseline margin | median min |pair| along CPU64 suffix trajectory | median pair-sign changes |")
    emit("|---|---:|---:|---:|---:|")
    for group_name, group_rows in [("flip", flips), ("control", controls)]:
        emit(
            f"| {group_name} | {len(group_rows)} | "
            f"{fmt(median([as_float(r, 'ref_top2_margin') for r in group_rows]))} | "
            f"{fmt(median([as_float(r, 'min_abs_pair') for r in group_rows]))} | "
            f"{fmt(median([as_float(r, 'pair_sign_changes') for r in group_rows]))} |"
        )
    emit("")

    emit("## Interpretation gate")
    emit("")
    emit("This report classifies depth-wise fate trajectories and compares them with matched non-flip controls. Repeated topology is evidence of recurring decision dynamics only after validity passes and replication grows beyond isolated cases. The source-branch grouping is only a descriptive grouping of where the numerical perturbation was injected; an association between source branch and topology would be a hypothesis to replicate, not an operator-level mechanism claim. This report does not identify QK, attention, MLP, or any other operator as a mechanism. Large adjacent pair changes are not causal operator attributions.")

    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
        print(f"[xray][fate-atlas-analysis] report={args.report}")


if __name__ == "__main__":
    main()
