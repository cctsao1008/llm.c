#!/usr/bin/env python3
"""Analyze numerical_fate_landscape_probe CSV output.

This is intentionally a population-analysis tool. It does not perform causal
localization or infer a mechanism. It summarizes perturbation-family overlap,
shared-vs-specific vulnerability, decision margins, near-miss controls, and a
simple normalized decision displacement:

    rho = (m_ref - m_alt) / m_ref

where m_ref and m_alt are the pair margins for the same reference winner and
selected competitor in one perturbation family. For a clean top-1 crossing,
rho > 1 means the pair margin crossed zero.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_FAMILIES = ["l00-qkv", "l00-attproj", "l00-fc", "l00-fcproj"]


def as_int(row, key):
    return int(row[key])


def as_float(row, key):
    return float(row[key])


def finite(values):
    return [x for x in values if math.isfinite(x)]


def quantile(values, q):
    x = sorted(finite(values))
    if not x:
        return math.nan
    if len(x) == 1:
        return x[0]
    p = q * (len(x) - 1)
    lo = int(math.floor(p))
    hi = int(math.ceil(p))
    if lo == hi:
        return x[lo]
    w = p - lo
    return x[lo] * (1.0 - w) + x[hi] * w


def stats(values):
    x = finite(values)
    if not x:
        return {"n": 0, "min": math.nan, "q25": math.nan, "median": math.nan,
                "q75": math.nan, "max": math.nan, "mean": math.nan}
    return {
        "n": len(x),
        "min": min(x),
        "q25": quantile(x, 0.25),
        "median": statistics.median(x),
        "q75": quantile(x, 0.75),
        "max": max(x),
        "mean": statistics.fmean(x),
    }


def fmt(x):
    if isinstance(x, int):
        return str(x)
    if x is None or not math.isfinite(float(x)):
        return "nan"
    return f"{float(x):.9e}"


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
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


def main():
    ap = argparse.ArgumentParser(description="Analyze numerical fate landscape CSV")
    ap.add_argument("csv", help="CSV from numerical_fate_landscape_probe")
    ap.add_argument("--report", default=None, help="write Markdown report")
    ap.add_argument("--unique-csv", default=None, help="write one row per unique flipped position")
    args = ap.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        raise SystemExit("empty CSV")

    families = [f for f in DEFAULT_FAMILIES if any(r["family"] == f for r in rows)]
    extra = sorted({r["family"] for r in rows} - set(families))
    families += extra

    # Position identity is global scan_index; keep batch/b/t as diagnostics.
    by_pos = defaultdict(dict)
    repeat_fail = 0
    for r in rows:
        key = as_int(r, "scan_index")
        fam = r["family"]
        if fam in by_pos[key]:
            raise SystemExit(f"duplicate row for scan_index={key}, family={fam}")
        by_pos[key][fam] = r
        repeat_fail += (as_int(r, "repeat_exact") == 0)

    expected = set(families)
    incomplete = [k for k, v in by_pos.items() if set(v) != expected]
    if incomplete:
        raise SystemExit(f"{len(incomplete)} positions do not contain exactly the expected families")

    # Baseline consistency across family rows is a cheap analysis-time validity gate.
    baseline_inconsistent = 0
    for fam_rows in by_pos.values():
        baseline_tuples = {
            (
                as_int(r, "ref_top1"), as_int(r, "ref_runner"),
                as_float(r, "ref_top2_margin"), as_int(r, "input_token"),
                as_int(r, "batch"), as_int(r, "b"), as_int(r, "t"),
            )
            for r in fam_rows.values()
        }
        if len(baseline_tuples) != 1:
            baseline_inconsistent += 1

    flips_by_family = {
        fam: {k for k, rs in by_pos.items() if as_int(rs[fam], "flip") == 1}
        for fam in families
    }
    union_flips = set().union(*flips_by_family.values()) if families else set()

    membership = {}
    for k in sorted(union_flips):
        membership[k] = tuple(f for f in families if k in flips_by_family[f])
    multiplicity = Counter(len(v) for v in membership.values())

    lines = []
    emit = lines.append
    emit("# Numerical Fate Landscape Analysis")
    emit("")
    emit(f"Input: `{args.csv}`")
    emit("")
    emit("## Validity")
    emit("")
    emit(f"- CSV rows: {len(rows)}")
    emit(f"- Unique positions: {len(by_pos)}")
    emit(f"- Families: {len(families)} ({', '.join(families)})")
    emit(f"- `repeat_exact=0` rows: {repeat_fail}")
    emit(f"- Baseline-inconsistent positions across families: {baseline_inconsistent}")
    emit(f"- Analysis validity: {1 if repeat_fail == 0 and baseline_inconsistent == 0 else 0}")
    emit("")

    emit("## Pairwise flip overlap")
    emit("")
    emit("Counts:")
    emit("")
    emit("| family | " + " | ".join(families) + " |")
    emit("|---|" + "---:|" * len(families))
    for a in families:
        vals = [len(flips_by_family[a] & flips_by_family[b]) for b in families]
        emit("| " + a + " | " + " | ".join(str(v) for v in vals) + " |")
    emit("")
    emit("Jaccard index:")
    emit("")
    emit("| family | " + " | ".join(families) + " |")
    emit("|---|" + "---:|" * len(families))
    for a in families:
        vals = []
        for b in families:
            u = flips_by_family[a] | flips_by_family[b]
            j = len(flips_by_family[a] & flips_by_family[b]) / len(u) if u else math.nan
            vals.append(j)
        emit("| " + a + " | " + " | ".join(f"{v:.3f}" if math.isfinite(v) else "nan" for v in vals) + " |")
    emit("")

    emit("## Flip multiplicity")
    emit("")
    emit("| number of perturbation families flipping a position | positions | fraction of union |")
    emit("|---:|---:|---:|")
    for n in range(1, len(families) + 1):
        c = multiplicity[n]
        frac = c / len(union_flips) if union_flips else math.nan
        emit(f"| {n} | {c} | {frac:.3f} |")
    emit("")

    # Reference top2 margin is baseline-invariant across family rows.
    margin_by_pos = {
        k: as_float(next(iter(rs.values())), "ref_top2_margin")
        for k, rs in by_pos.items()
    }
    emit("## Multiplicity vs baseline top-2 margin")
    emit("")
    emit("| multiplicity | n | min | Q25 | median | Q75 | max |")
    emit("|---:|---:|---:|---:|---:|---:|---:|")
    for n in range(1, len(families) + 1):
        s = stats([margin_by_pos[k] for k, m in membership.items() if len(m) == n])
        emit(f"| {n} | {s['n']} | {fmt(s['min'])} | {fmt(s['q25'])} | {fmt(s['median'])} | {fmt(s['q75'])} | {fmt(s['max'])} |")
    emit("")

    emit("## Flip vs near-miss baseline margins")
    emit("")
    emit("Near-miss means `near_ref=1` and `flip=0` for that perturbation family.")
    emit("")
    emit("| family | group | n | Q25 | median | Q75 |")
    emit("|---|---|---:|---:|---:|---:|")
    for fam in families:
        fam_rows = [rs[fam] for rs in by_pos.values()]
        groups = {
            "flip": [as_float(r, "ref_top2_margin") for r in fam_rows if as_int(r, "flip") == 1],
            "near-miss": [as_float(r, "ref_top2_margin") for r in fam_rows if as_int(r, "flip") == 0 and as_int(r, "near_ref") == 1],
        }
        for name, vals in groups.items():
            s = stats(vals)
            emit(f"| {fam} | {name} | {s['n']} | {fmt(s['q25'])} | {fmt(s['median'])} | {fmt(s['q75'])} |")
    emit("")

    emit("## Normalized decision displacement rho")
    emit("")
    emit(r"Definition: `rho = (ref_pair_margin - alt_pair_margin) / ref_pair_margin`.")
    emit("For a positive reference pair margin, `rho > 1` means that pair crossed zero.")
    emit("")
    emit("| family | group | n | Q25 | median | Q75 | max | P(rho>1) |")
    emit("|---|---|---:|---:|---:|---:|---:|---:|")
    for fam in families:
        fam_rows = [rs[fam] for rs in by_pos.values()]
        for group_name, pred in [
            ("flip", lambda r: as_int(r, "flip") == 1),
            ("near-miss", lambda r: as_int(r, "flip") == 0 and as_int(r, "near_ref") == 1),
        ]:
            vals = []
            for r in fam_rows:
                if not pred(r):
                    continue
                m0 = as_float(r, "ref_pair_margin")
                m1 = as_float(r, "alt_pair_margin")
                if m0 != 0.0:
                    vals.append((m0 - m1) / m0)
            s = stats(vals)
            p = sum(v > 1.0 for v in vals) / len(vals) if vals else math.nan
            emit(f"| {fam} | {group_name} | {s['n']} | {fmt(s['q25'])} | {fmt(s['median'])} | {fmt(s['q75'])} | {fmt(s['max'])} | {p:.3f} |")
    emit("")

    emit("## Unique flipped positions")
    emit("")
    emit("| scan_index | batch | b | t | input | ref_top1 | ref_runner | ref_margin | multiplicity | families |")
    emit("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    unique_records = []
    for k in sorted(union_flips):
        first = next(iter(by_pos[k].values()))
        fams = membership[k]
        rec = {
            "scan_index": k,
            "batch": as_int(first, "batch"),
            "b": as_int(first, "b"),
            "t": as_int(first, "t"),
            "input_token": as_int(first, "input_token"),
            "ref_top1": as_int(first, "ref_top1"),
            "ref_runner": as_int(first, "ref_runner"),
            "ref_top2_margin": as_float(first, "ref_top2_margin"),
            "multiplicity": len(fams),
            "families": ";".join(fams),
        }
        unique_records.append(rec)
        emit(f"| {k} | {rec['batch']} | {rec['b']} | {rec['t']} | {rec['input_token']} | {rec['ref_top1']} | {rec['ref_runner']} | {fmt(rec['ref_top2_margin'])} | {rec['multiplicity']} | {rec['families']} |")
    emit("")

    emit("## All-family shared vulnerabilities")
    emit("")
    all_shared = [r for r in unique_records if r["multiplicity"] == len(families)]
    if all_shared:
        for r in all_shared:
            emit(f"- scan_index={r['scan_index']} batch={r['batch']} b={r['b']} t={r['t']} ref_margin={fmt(r['ref_top2_margin'])} ref={r['ref_top1']} runner={r['ref_runner']}")
    else:
        emit("- none")
    emit("")

    emit("## Single-family vulnerabilities")
    emit("")
    single_counts = Counter(r["families"] for r in unique_records if r["multiplicity"] == 1)
    for fam in families:
        emit(f"- {fam}: {single_counts[fam]}")
    emit("")

    emit("## Interpretation gate")
    emit("")
    emit("This report describes population-level final-decision geometry. It does **not** identify a causal layer, operator, numerical mechanism, or architecture-general law. Use it to select representative flip and near-miss cases for subsequent exact/high-precision localization.")

    text = "\n".join(lines) + "\n"
    print(text, end="")

    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
        print(f"[xray][fate-landscape-analysis] report={args.report}")

    if args.unique_csv:
        with open(args.unique_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["scan_index", "batch", "b", "t", "input_token", "ref_top1",
                          "ref_runner", "ref_top2_margin", "multiplicity", "families"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(unique_records)
        print(f"[xray][fate-landscape-analysis] unique_csv={args.unique_csv}")


if __name__ == "__main__":
    main()
