"""scripts/build_reports.py — Aggregate v3 experimental metrics into CSV and inject Markdown tables.

Reads:
  experiments/v3/**/metrics.json
Writes:
  experiments/v3/results_master.csv
Renders tables between markers:
  <!-- AUTO:<table_name> -->
  ...
  <!-- /AUTO -->
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def bootstrap_ci_95(values: list[float], n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    """Calculate 95% non-parametric bootstrap confidence interval."""
    if len(values) == 1:
        return values[0], values[0]
    rng = np.random.RandomState(seed)
    boot_means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def aggregate_v3_results() -> pd.DataFrame:
    exp_v3 = Path("experiments/v3")
    if not exp_v3.exists():
        return pd.DataFrame()

    records = []
    # Discover metric files by group directory
    for group_dir in exp_v3.iterdir():
        if not group_dir.is_dir() or group_dir.name.startswith((".", "_")):
            continue

        seed_metric_files = list(group_dir.glob("seed*/metrics.json"))
        if not seed_metric_files:
            continue

        accuracies = []
        macro_f1s = []
        cb_f1s = []
        params_str = "N/A"

        for mf in seed_metric_files:
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
                accuracies.append(float(data.get("accuracy", 0.0)))
                macro_f1s.append(float(data.get("f1_macro", data.get("val_f1_macro", 0.0))))
                cb_f1s.append(float(data.get("f1_clickbait", 0.0)))

                if "params" in data and isinstance(data["params"], (int, float)):
                    p_val = int(data["params"])
                    params_str = f"{p_val / 1e6:.2f}M" if p_val >= 1_000_000 else f"{p_val:,}"
                else:
                    cfg_file = mf.parent / "config.yaml"
                    if cfg_file.exists():
                        cfg_text = cfg_file.read_text(encoding="utf-8")
                        if "7.42M" in cfg_text:
                            params_str = "7.42M"
                        elif "110.6M" in cfg_text:
                            params_str = "110.6M"
                        elif "TF-IDF" in cfg_text:
                            params_str = "Linear"
            except Exception as e:
                logger.warning("Error reading %s: %s", mf, e)

        if not macro_f1s:
            continue

        f1_lo, f1_hi = bootstrap_ci_95(macro_f1s)
        records.append({
            "group": group_dir.name,
            "n_seeds": len(macro_f1s),
            "params": params_str,
            "acc_mean": round(float(np.mean(accuracies)), 4),
            "acc_std": round(float(np.std(accuracies)), 4),
            "f1_mean": round(float(np.mean(macro_f1s)), 4),
            "f1_std": round(float(np.std(macro_f1s)), 4),
            "f1_ci_lo": round(f1_lo, 4),
            "f1_ci_hi": round(f1_hi, 4),
            "cb_f1_mean": round(float(np.mean(cb_f1s)), 4),
            "latency_ms": "TBD",
        })

    df = pd.DataFrame(records)
    if not df.empty:
        out_csv = exp_v3 / "results_master.csv"
        df.to_csv(out_csv, index=False)
        logger.info("Saved %d group summaries to %s", len(df), out_csv)
    return df


def update_auto_block(doc_path: Path, table_name: str, markdown_table: str) -> bool:
    """Inject markdown_table between <!-- AUTO:<table_name> --> markers."""
    if not doc_path.exists():
        return False
    content = doc_path.read_text(encoding="utf-8")
    pattern = rf"(<!-- AUTO:{re.escape(table_name)} -->\n)(?:.*?)(<!-- /AUTO -->)"
    replacement = rf"\g<1>{markdown_table}\n\g<2>"
    if re.search(pattern, content, flags=re.DOTALL):
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        doc_path.write_text(new_content, encoding="utf-8")
        logger.info("Updated AUTO table '%s' in %s", table_name, doc_path)
        return True
    return False


if __name__ == "__main__":
    aggregate_v3_results()
