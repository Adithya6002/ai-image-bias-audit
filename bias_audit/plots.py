"""Static figures for the report (results/figures/*.png)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import stats as st
from .data import MODELS, RACE_CATEGORIES, subjects, table

# Categorical slots 1-3 (validated all-pairs) for the three models; recessive ink for text and axes.
MODEL_COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a"]))
RACE_COLORS = dict(zip(RACE_CATEGORIES + ["Cannot say"],
                       ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#b8b7b2"]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, facecolor="white")
    plt.close(fig)


def female_share_vs_reference(e1: pd.DataFrame, out: Path) -> None:
    """B.1: pooled female share with exact 95% CI against the reference."""
    df = e1.iloc[::-1]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    ax.hlines(y, df.CP95_lo, df.CP95_hi, color=MODEL_COLORS[MODELS[0]], lw=2, label="95 % exact CI")
    ax.plot(df.female_share, y, "o", ms=8, color=MODEL_COLORS[MODELS[0]], mec="white", mew=1.5, label="Observed share")
    ax.plot(df.reference_female, y, "D", ms=7, mfc="white", mec=INK, mew=1.3, label="Reference")
    for yi, (_, r) in zip(y, df.iterrows()):
        if r["H0 (no bias)"] == "Rejected":
            ax.text(1.03, yi, "H₀ rejected", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, df.subject)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Female-presenting share (all models pooled)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
    _save(fig, out, "B1_female_share_vs_reference.png")


def share_by_model(counts: pd.DataFrame, attribute: str, category: str, out: Path, name: str, xlabel: str) -> None:
    """B.2 / B.3: per-model share of one category, per prompt."""
    subs = subjects(counts)[::-1]
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    offsets = dict(zip(MODELS, [0.22, 0.0, -0.22]))
    for m in MODELS:
        shares = []
        for s in subs:
            t = table(counts, s, attribute)
            row = t.loc[m]
            n = row.drop("Cannot say", errors="ignore").sum() if attribute == "Race" else row.sum()
            shares.append(row.get(category, 0) / n)
        ax.plot(shares, np.arange(len(subs)) + offsets[m], "o", ms=8, color=MODEL_COLORS[m], mec="white", mew=1.5, label=m)
    ax.set_yticks(np.arange(len(subs)), subs)
    ax.set_xlim(-0.03, 1.03)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9)
    _save(fig, out, name)


def race_composition(counts: pd.DataFrame, out: Path) -> None:
    """B.4: perceived-race composition of all images, per model (stacked, 2px gaps)."""
    cats = RACE_CATEGORIES + ["Cannot say"]
    race = counts[counts.attribute == "Race"]
    totals = race.pivot_table(index="model", columns="category", values="count", aggfunc="sum").reindex(MODELS)
    totals = totals.reindex(columns=cats, fill_value=0).fillna(0)
    share = totals.div(totals.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    left = np.zeros(len(MODELS))
    y = np.arange(len(MODELS))[::-1]
    for c in cats:
        ax.barh(y, share[c], left=left, color=RACE_COLORS[c], edgecolor="white", linewidth=2, height=0.6, label=c)
        for yi, l, w in zip(y, left, share[c]):
            if w >= 0.08:
                ax.text(l + w / 2, yi, f"{w:.0%}", ha="center", va="center", fontsize=8, color="white" if c != "Cannot say" else INK)
        left += share[c].to_numpy()
    ax.set_yticks(y, MODELS)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of all images (45 per model)")
    ax.grid(visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=6, fontsize=8, handlelength=1)
    _save(fig, out, "B4_race_composition_by_model.png")


def write_figures(counts: pd.DataFrame, e1: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    female_share_vs_reference(e1, out)
    share_by_model(counts, "Sex", "Female", out, "B2_female_share_by_model.png",
                   "Female-presenting share per model (5 images each)")
    share_by_model(counts, "Race", "White", out, "B3_white_share_by_model.png",
                   "White share per model (5 images each; 'Cannot say' excluded)")
    race_composition(counts, out)
