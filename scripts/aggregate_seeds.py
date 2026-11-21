"""
scripts/aggregate_seeds.py — Aggregate multi-seed runs into mean ± std tables with bootstrap CIs.

Groups run directories named  <group>_s<seed>  under experiments/ and reports, per group:
  - mean ± std over seeds of test Accuracy / Macro-F1 / Clickbait-F1 (threshold 0.50 and calibrated)
  - pooled-seed bootstrap 95% CI of Macro-F1 (per seed, then averaged range)
Optionally compares two groups with a paired bootstrap on seed-matched predictions.

Usage:
    python -m scripts.aggregate_seeds --groups student_small_pretrained_distilled student_teacher_init_distilled
    python -m scripts.aggregate_seeds --groups A B --compare A B
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.evaluation.stats import bootstrap_ci, holm_correction, mean_std, paired_bootstrap


def find_runs(exp_dir: Path, group: str) -> dict[int, Path]:
    pat = re.compile(rf"^{re.escape(group)}_s(\d+)$")
    runs = {}
    for d in exp_dir.iterdir():
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if m and (d / "metrics.json").exists():
            runs[int(m.group(1))] = d
    # Fallback: include canonical single-seed directory (seed=42) if present
    base_dir = exp_dir / group
    if 42 not in runs and (base_dir / "metrics.json").exists():
        runs[42] = base_dir
    return dict(sorted(runs.items()))


def summarize(exp_dir: Path, group: str) -> dict | None:
    runs = find_runs(exp_dir, group)
    if not runs:
        return None
    rows = {k: [] for k in ("acc", "f1", "f1_cb", "cal_acc", "cal_f1", "ci_lo", "ci_hi")}
    for seed, d in runs.items():
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        rows["acc"].append(m["accuracy"])
        rows["f1"].append(m["f1_macro"])
        rows["f1_cb"].append(m["f1_clickbait"])
        cal = m.get("calibrated", m)
        rows["cal_acc"].append(cal["accuracy"])
        rows["cal_f1"].append(cal["f1_macro"])
        lg, lb = d / "test_logits.npy", d / "test_labels.npy"
        if lg.exists() and lb.exists():
            _, lo, hi = bootstrap_ci(np.load(lb), np.load(lg).argmax(1))
            rows["ci_lo"].append(lo)
            rows["ci_hi"].append(hi)
    out = {"group": group, "seeds": list(runs.keys()), "n": len(runs)}
    for k, v in rows.items():
        if v:
            out[k] = mean_std(v)
    return out


def fmt(ms) -> str:
    return f"{ms[0]:.4f} ± {ms[1]:.4f}" if ms else "n/a"


def compare(exp_dir: Path, a: str, b: str) -> list[dict]:
    ra, rb = find_runs(exp_dir, a), find_runs(exp_dir, b)
    out = []
    for seed in sorted(set(ra) & set(rb)):
        try:
            y = np.load(ra[seed] / "test_labels.npy")
            pa = np.load(ra[seed] / "test_logits.npy").argmax(1)
            pb = np.load(rb[seed] / "test_logits.npy").argmax(1)
        except FileNotFoundError:
            continue
        r = paired_bootstrap(y, pa, pb)
        r["seed"] = seed
        out.append(r)
    if out:
        adj = holm_correction([r["p_value"] for r in out])
        for r, p in zip(out, adj):
            r["p_holm"] = p
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--groups", nargs="+", required=True)
    p.add_argument("--exp-dir", default="experiments")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    p.add_argument("--output", default="experiments/multiseed_summary.md")
    args = p.parse_args(argv)

    exp_dir = Path(args.exp_dir)
    lines = [
        "# Multi-seed & Bootstrap Summary (test split, mean ± sample std over seeds)",
        "",
        "| Group | #Seeds | Acc | Macro-F1 | 95% Bootstrap CI (F1) | Clickbait-F1 | Acc (cal) | Macro-F1 (cal) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for g in args.groups:
        s = summarize(exp_dir, g)
        if not s:
            lines.append(f"| {g} | 0 | no runs found | | | | | |")
            continue
        ci_str = (
            f"[{s['ci_lo'][0]:.4f}, {s['ci_hi'][0]:.4f}]"
            if "ci_lo" in s and "ci_hi" in s
            else "n/a"
        )
        lines.append(
            f"| {g} | {s['n']} | {fmt(s.get('acc'))} | {fmt(s.get('f1'))} | {ci_str} | {fmt(s.get('f1_cb'))} "
            f"| {fmt(s.get('cal_acc'))} | {fmt(s.get('cal_f1'))} |"
        )

    if args.compare:
        a, b = args.compare
        res = compare(exp_dir, a, b)
        lines += ["", f"## Paired bootstrap: {a} − {b} (Macro-F1, seed-matched)", "",
                  "| Seed | ΔF1 | 95% CI | p | p (Holm) |", "|:---:|:---:|:---:|:---:|:---:|"]
        for r in res:
            lines.append(f"| {r['seed']} | {r['delta']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['p_value']:.4f} | {r['p_holm']:.4f} |")
        if not res:
            lines.append("| no seed-matched runs with saved logits | | | | |")

    text = "\n".join(lines)
    Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
