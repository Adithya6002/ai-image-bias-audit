"""Run every required experiment and write results/*.csv plus results/REPORT.md.

Usage:  uv run python run.py [path/to/Aggregation.xlsx]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bias_audit import analysis as an
from bias_audit.data import load_counts
from bias_audit.report import write_report

SEED = 20261029
ROOT = Path(__file__).parent
OUT = ROOT / "results"


def main() -> None:
    xlsx = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "Aggregation.xlsx"
    rng = np.random.default_rng(SEED)
    counts, bench = load_counts(xlsx)

    res: dict[str, pd.DataFrame] = {}
    res["e1_gender_vs_reference_pooled"], res["e1_gender_vs_reference_per_model"] = \
        an.gender_vs_reference(counts, bench, rng)
    res["e2_gender_model_homogeneity"] = an.model_homogeneity(counts, "Sex", rng)
    res["e3_occupation_gender_association"] = an.occupation_gender_association(counts, rng)
    res["e4_race_vs_equal_pooled"], res["e4_race_vs_equal_per_model"] = an.race_vs_uniform(counts, rng)
    res["e5_race_model_homogeneity"] = an.model_homogeneity(counts, "Race", rng)

    res["r1_annotator_cell_agreement"], res["r1_annotator_retest"], res["r1_annotator_occupation"] = \
        an.annotator_robustness(counts, bench, rng)
    res["r2_benchmark_sensitivity"] = an.benchmark_sensitivity(counts, len(res["e1_gender_vs_reference_pooled"]))
    res["r3_leave_one_model_out"] = an.leave_one_model_out(counts, bench).reset_index()
    res["r4_fragility"] = an.fragility_table(res["e1_gender_vs_reference_pooled"])
    res["r5_ambiguous_labels"] = an.ambiguous_label_sensitivity(counts, bench)

    OUT.mkdir(exist_ok=True)
    counts.to_csv(OUT / "input_counts_long.csv", index=False)
    bench.to_csv(OUT / "input_benchmarks.csv", index=False)
    for name, df in res.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    write_report(res, bench, OUT / "REPORT.md", seed=SEED)
    print(f"Wrote {len(res)} result tables and REPORT.md to {OUT}")


if __name__ == "__main__":
    main()
