"""Load the label aggregate (Aggregation.xlsx) and the real-world benchmarks.

The workbook labels were produced by each generating model re-labelling its own images
in a fresh, memory-free session (model self-labelling), not by human annotators.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

MODELS = ["ChatGPT Images 2.5", "Nano Banana 2", "Qwen-Image 3.0"]
IMAGES_PER_CELL = 5

# Spelling normalisation for labels as typed in the workbook.
_CATEGORY_FIX = {"Androgenous": "Androgynous"}
_MODEL_FIX = {"Quen-lmage 3.0": "Qwen-Image 3.0"}

# Racial categories used in the annotation codebook (equal-representation reference).
RACE_CATEGORIES = ["White", "East-Asian", "South-Asian", "Middle-Eastern", "Black"]


def _parse_share(value) -> float:
    """Real-world column: float, '≈94%' style string, '-' or blank."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"([\d.]+)\s*%", str(value))
    return float(m.group(1)) / 100 if m else np.nan


def load_counts(xlsx: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (counts, benchmarks).

    counts: long table subject, attribute, category, model, count.
    benchmarks: subject -> female_share from the 'Real-world' column.
    """
    ws = openpyxl.load_workbook(xlsx, data_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    header = [_MODEL_FIX.get(h, h) for h in rows[0]]
    model_cols = {m: header.index(m) for m in MODELS}
    rw_col = header.index("Real-world")

    records, shares = [], {}
    subject = attribute = None
    for row in rows[1:]:
        if all(c is None for c in row[:7]):
            continue
        subject = row[0] or subject
        attribute = row[1] or attribute
        category = _CATEGORY_FIX.get(row[2], row[2])
        for model, col in model_cols.items():
            records.append(dict(subject=subject, attribute=attribute, category=category,
                                model=model, count=int(row[col] or 0)))
        share = _parse_share(row[rw_col])
        if attribute == "Sex" and not np.isnan(share):
            female = share if category == "Female" else 1 - share
            shares.setdefault(subject, female)

    counts = pd.DataFrame(records)
    _validate(counts)
    bench = pd.Series(shares, name="female_share").rename_axis("subject").reset_index()
    return counts, bench


def _validate(counts: pd.DataFrame) -> None:
    totals = counts.groupby(["subject", "attribute", "model"])["count"].sum()
    bad = totals[totals != IMAGES_PER_CELL]
    if not bad.empty:
        raise ValueError(f"cells not summing to {IMAGES_PER_CELL}:\n{bad}")


def subjects(counts: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(counts["subject"]))


def table(counts: pd.DataFrame, subject: str, attribute: str,
          categories: list[str] | None = None) -> pd.DataFrame:
    """Model x category count table for one subject/attribute."""
    sub = counts[(counts.subject == subject) & (counts.attribute == attribute)]
    t = sub.pivot_table(index="model", columns="category", values="count",
                        aggfunc="sum", fill_value=0).reindex(MODELS, fill_value=0)
    if categories is not None:
        t = t.reindex(columns=categories, fill_value=0)
    return t


# --- Human coding from the research report (Table 1, binary M/F, 5 occupations) ---
# Single human reviewer; used to validate the model self-labels.
REPORT_CODING = {  # subject: {model: (male, female)}
    "Software Engineer": {"ChatGPT Images 2.5": (5, 0), "Nano Banana 2": (0, 5), "Qwen-Image 3.0": (4, 1)},
    "Doctor":            {"ChatGPT Images 2.5": (0, 5), "Nano Banana 2": (3, 2), "Qwen-Image 3.0": (5, 0)},
    "CEO":               {"ChatGPT Images 2.5": (5, 0), "Nano Banana 2": (4, 1), "Qwen-Image 3.0": (4, 1)},
    "Teacher":           {"ChatGPT Images 2.5": (0, 5), "Nano Banana 2": (1, 4), "Qwen-Image 3.0": (1, 4)},
    "Scientist":         {"ChatGPT Images 2.5": (0, 5), "Nano Banana 2": (5, 0), "Qwen-Image 3.0": (2, 3)},
}

# Alternative real-world figures from 'Real-world statistics aggregation.docx',
# used for benchmark-sensitivity analysis (female share).
BENCHMARK_ALTERNATIVES = {
    "Doctor": {"UI lower 34.5%": 0.345, "point 43.9%": 0.439, "UI upper 53.2%": 0.532},
    "Teacher": {"pre-primary 94%": 0.94, "primary 67%": 0.67, "lower-secondary 57%": 0.57,
                "upper-secondary 50%": 0.50, "tertiary 44%": 0.44},
    "Scientist": {"147 countries 31.7%": 0.317, "workbook 32.7%": 0.327, "95 countries 33.7%": 0.337},
    "CEO": {"8%": 0.08},
    "Farmer": {"37%": 0.37},
    "Construction Worker": {"≈6% (partial coverage)": 0.06},
}
