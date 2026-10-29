"""The 'required experiments': significance tests with multiplicity correction,
bootstrap CIs, effect sizes, and robustness checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import stats as st
from .data import (BENCHMARK_ALTERNATIVES, MODELS, RACE_CATEGORIES, REPORT_CODING,
                   subjects, table)

ALPHA = 0.05
POOLED = "Pooled (all 3 models)"
OCCUPATIONS = ["Software Engineer", "Doctor", "CEO", "Teacher",
               "Construction Worker", "Scientist", "Farmer"]


def _decide(df: pd.DataFrame, p_col: str = "p") -> pd.DataFrame:
    df["p_holm"] = st.adjust(df[p_col], "holm")
    df["q_bh"] = st.adjust(df[p_col], "fdr_bh")
    df["H0 (no bias)"] = np.where(df["p_holm"] < ALPHA, "Rejected", "Not rejected")
    return df


# Prompts with no defensible gender reference: excluded from every test that needs one.
# Software Engineer: the only figures found (<50% female) were not from academic sources,
# and parity is clearly not appropriate for this occupation.
NO_GENDER_REFERENCE = {"Software Engineer"}


def _gender_reference(subject: str, bench: pd.DataFrame) -> tuple[float, str] | None:
    """Real-world female share, 50% parity for non-occupational prompts, or None if excluded."""
    if subject in NO_GENDER_REFERENCE:
        return None
    row = bench[bench.subject == subject]
    if row.empty:
        return 0.5, "parity (no real-world figure)"
    return float(row.female_share.iloc[0]), "real-world"


def reference_subjects(counts, bench) -> list[str]:
    return [s for s in subjects(counts) if _gender_reference(s, bench) is not None]


def _binary_sex(counts: pd.DataFrame, subject: str) -> pd.DataFrame:
    return table(counts, subject, "Sex", ["Male", "Female"])


def _proportion_row(subject, scope, female, n, ref, ref_src, rng, boot_k, boot_n):
    share = female / n
    lo, hi = st.clopper_pearson(female, n)
    blo, bhi = st.bootstrap_share(boot_k, boot_n, rng)
    dir_ = min(share / ref, (1 - share) / (1 - ref))
    return {
        "subject": subject, "scope": scope, "n": n, "female": female,
        "female_share": share, "CP95_lo": lo, "CP95_hi": hi, "boot95_lo": blo, "boot95_hi": bhi,
        "reference_female": ref, "reference": ref_src,
        "cohen_h": st.cohen_h(share, ref),
        "disparate_impact_ratio": dir_, "fails_4/5_rule": dir_ < 0.8,
        "p": st.binom_test(female, n, ref), "min_attainable_p": st.min_attainable_p(n, ref),
    }


# ---------- Experiment 1: perceived gender vs reference distribution ----------

def gender_vs_reference(counts, bench, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H0: P(female-presenting) equals the reference (real-world share, else 50%).

    Returns two Holm families: pooled per prompt (primary) and per model x prompt.
    """
    pooled, per_model = [], []
    for subj in reference_subjects(counts, bench):
        t = _binary_sex(counts, subj)
        ref, src = _gender_reference(subj, bench)
        k, n = t["Female"].to_numpy(), t.sum(1).to_numpy()
        pooled.append(_proportion_row(subj, POOLED, int(k.sum()), int(n.sum()), ref, src, rng, k, n))
        for m, ki, ni in zip(MODELS, k, n):
            per_model.append(_proportion_row(subj, m, int(ki), int(ni), ref, src, rng, [ki], [ni]))
    return _decide(pd.DataFrame(pooled)), _decide(pd.DataFrame(per_model))


# ---------- Experiment 2: do the models differ? (gender) ----------

def model_homogeneity(counts, attribute: str, rng) -> pd.DataFrame:
    """H0: the perceived-attribute distribution is the same for all three models."""
    rows = []
    for subj in subjects(counts):
        t = table(counts, subj, attribute)
        chi2, p = st.permutation_independence(t.to_numpy(), rng)
        vlo, vhi = st.bootstrap_cramers_v(t.to_numpy(), rng)
        focus = "Female" if attribute == "Sex" else "White"
        k = t[focus].to_numpy() if focus in t else np.zeros(len(MODELS), int)
        gap, glo, ghi = st.bootstrap_gap(k, t.sum(1).to_numpy(), rng)
        rows.append({
            "subject": subj, "attribute": attribute,
            "table (rows=models)": "; ".join(
                f"{m.split()[0]} " + ",".join(f"{c}={v}" for c, v in r.items() if v)
                for m, r in t.iterrows()),
            "chi2": chi2, "p": p, "cramers_v": st.cramers_v(t.to_numpy()),
            "V_boot95_lo": vlo, "V_boot95_hi": vhi,
            f"parity_gap_{focus.lower()}_share": gap, "gap_boot95_lo": glo, "gap_boot95_hi": ghi,
        })
    return _decide(pd.DataFrame(rows))


# ---------- Experiment 3: is perceived gender independent of occupation? ----------

def occupation_gender_association(counts, rng, coding: dict | None = None,
                                  occupations=OCCUPATIONS) -> pd.DataFrame:
    """H0: perceived gender is independent of the occupation named in the prompt."""
    def occ_table(models):
        rows = []
        for occ in occupations:
            if coding is not None:
                rows.append([sum(coding[occ][m][i] for m in models) for i in (0, 1)])
            else:
                t = _binary_sex(counts, occ).loc[models]
                rows.append([t["Male"].sum(), t["Female"].sum()])
        return np.array(rows)

    out = []
    for scope, models in [(POOLED, MODELS)] + [(m, [m]) for m in MODELS]:
        t = occ_table(models)
        chi2, p = st.permutation_independence(t, rng)
        vlo, vhi = st.bootstrap_cramers_v(t, rng)
        tt = st.drop_empty(t)
        asym = sps.chi2_contingency(tt, correction=False) if min(tt.shape) > 1 else None
        out.append({"scope": scope, "n": int(t.sum()), "chi2": chi2,
                    "df": (tt.shape[0] - 1) * (tt.shape[1] - 1),
                    "p": p, "p_asymptotic": asym.pvalue if asym else np.nan,
                    "cramers_v": st.cramers_v(t), "V_boot95_lo": vlo, "V_boot95_hi": vhi})
    return _decide(pd.DataFrame(out))


# ---------- Experiment 4: perceived race vs equal representation ----------

def race_vs_uniform(counts, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H0: perceived race is equally distributed over the 5 codebook categories.

    'Cannot say' labels are excluded. No real-world race benchmark was supplied, so this
    is an equal-representation (diversity) reference, not an occupational-labour reference.
    """
    probs = np.full(len(RACE_CATEGORIES), 1 / len(RACE_CATEGORIES))

    def row(subj, scope, vec, boot_k, boot_n):
        n = int(vec.sum())
        chi2, p = st.multinomial_gof(vec, probs, rng)
        white = int(vec[0])
        lo, hi = st.clopper_pearson(white, n)
        blo, bhi = st.bootstrap_share(boot_k, boot_n, rng)
        return {"subject": subj, "scope": scope, "n": n,
                "counts": ", ".join(f"{c}={v}" for c, v in zip(RACE_CATEGORIES, vec) if v),
                "white_share": white / n, "CP95_lo": lo, "CP95_hi": hi,
                "boot95_lo": blo, "boot95_hi": bhi,
                "white_vs_20%_ratio": (white / n) / 0.2,
                "cohen_w": st.cohen_w(vec, probs), "chi2": chi2, "p": p}

    pooled, per_model = [], []
    for subj in subjects(counts):
        t = table(counts, subj, "Race", RACE_CATEGORIES)
        w, n = t["White"].to_numpy(), t.sum(1).to_numpy()
        pooled.append(row(subj, POOLED, t.sum(0).to_numpy(), w, n))
        for m in MODELS:
            per_model.append(row(subj, m, t.loc[m].to_numpy(), [t.loc[m, "White"]], [t.loc[m].sum()]))
    return _decide(pd.DataFrame(pooled)), _decide(pd.DataFrame(per_model))


# ---------- Robustness ----------

def annotator_robustness(counts, bench, rng) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Re-run the gender tests on the research report's independent coding of 5 occupations."""
    occs = list(REPORT_CODING)
    agree = []
    for occ in occs:
        t = _binary_sex(counts, occ)
        for m in MODELS:
            a = (int(t.loc[m, "Male"]), int(t.loc[m, "Female"]))
            b = REPORT_CODING[occ][m]
            agree.append({"subject": occ, "model": m, "workbook M/F": f"{a[0]}/{a[1]}",
                          "report M/F": f"{b[0]}/{b[1]}", "identical": a == b,
                          "abs diff (female)": abs(a[1] - b[1])})
    agree = pd.DataFrame(agree)

    rows = []
    for occ in occs:
        ref = (_gender_reference(occ, bench) or (np.nan, ""))[0]
        for source in ("workbook", "report"):
            if source == "workbook":
                t = _binary_sex(counts, occ)
                k, n = t["Female"].to_numpy(), t.sum(1).to_numpy()
                mtab = t.to_numpy()
            else:
                k = np.array([REPORT_CODING[occ][m][1] for m in MODELS])
                n = np.array([sum(REPORT_CODING[occ][m]) for m in MODELS])
                mtab = np.array([REPORT_CODING[occ][m] for m in MODELS])
            _, p_h = st.permutation_independence(mtab, rng)
            rows.append({"subject": occ, "coding": source, "female": int(k.sum()), "n": int(n.sum()),
                         "reference_female": ref, "p_vs_reference": np.nan if np.isnan(ref) else st.binom_test(int(k.sum()), int(n.sum()), ref),
                         "p_model_homogeneity": p_h})
    df = pd.DataFrame(rows)
    for col in ("p_vs_reference", "p_model_homogeneity"):
        for source in ("workbook", "report"):
            mask = (df.coding == source) & df[col].notna()
            df.loc[mask, col + "_holm"] = st.adjust(df.loc[mask, col], "holm")
    for col in ("p_vs_reference", "p_model_homogeneity"):
        df[col.replace("p_", "reject_")] = np.where(df[col + "_holm"].notna(),
                                                     df[col + "_holm"] < ALPHA, None)

    occ_wb = occupation_gender_association(counts, rng, occupations=occs).iloc[[0]]
    occ_rep = occupation_gender_association(counts, rng, coding=REPORT_CODING, occupations=occs).iloc[[0]]
    occ = pd.concat([occ_wb.assign(coding="workbook"), occ_rep.assign(coding="report")])
    return agree, df, occ


def benchmark_sensitivity(counts, n_tests: int) -> pd.DataFrame:
    """Pooled E1 under alternative references; Bonferroni level alpha / n_tests."""
    rows = []
    for subj, alts in BENCHMARK_ALTERNATIVES.items():
        t = _binary_sex(counts, subj)
        k, n = int(t["Female"].sum()), int(t.to_numpy().sum())
        for label, ref in list(alts.items()) + [("parity 50%", 0.5)]:
            p = st.binom_test(k, n, ref)
            rows.append({"subject": subj, "reference": label, "reference_female": ref,
                         "female": k, "n": n, "female_share": k / n,
                         "cohen_h": st.cohen_h(k / n, ref), "p": p,
                         "significant_bonferroni": p < ALPHA / n_tests})
    return pd.DataFrame(rows)


def leave_one_model_out(counts, bench) -> pd.DataFrame:
    rows = []
    subs = reference_subjects(counts, bench)
    for dropped in [None] + MODELS:
        keep = [m for m in MODELS if m != dropped]
        ps = []
        for subj in subs:
            t = _binary_sex(counts, subj).loc[keep]
            ref, _ = _gender_reference(subj, bench)
            ps.append(st.binom_test(int(t["Female"].sum()), int(t.to_numpy().sum()), ref))
        holm = st.adjust(ps, "holm")
        for subj, p, ph in zip(subs, ps, holm):
            rows.append({"dropped_model": dropped or "none", "subject": subj, "p": p,
                         "p_holm": ph, "rejected": ph < ALPHA})
    df = pd.DataFrame(rows)
    wide = df.pivot(index="subject", columns="dropped_model", values="rejected").reindex(subs)
    wide["stable"] = wide.nunique(axis=1) == 1
    return wide[["none"] + MODELS + ["stable"]]


def fragility_table(pooled: pd.DataFrame) -> pd.DataFrame:
    rows, bonf = [], ALPHA / len(pooled)
    for _, r in pooled.iterrows():
        rows.append({"subject": r.subject, "female": r.female, "n": r.n,
                     "reference_female": r.reference_female, "p": r.p,
                     "flips_to_reverse_at_0.05": st.fragility(r.female, r.n, r.reference_female, ALPHA),
                     "flips_to_reverse_bonferroni": st.fragility(r.female, r.n, r.reference_female, bonf)})
    return pd.DataFrame(rows)


def ambiguous_label_sensitivity(counts, bench) -> pd.DataFrame:
    """Re-test prompts with 'Androgynous' labels counting them as female, then as male."""
    rows = []
    for subj in reference_subjects(counts, bench):
        t = table(counts, subj, "Sex")
        amb = int(t.drop(columns=["Male", "Female"], errors="ignore").to_numpy().sum())
        if not amb:
            continue
        ref, _ = _gender_reference(subj, bench)
        f, m = int(t["Female"].sum()), int(t["Male"].sum())
        for label, k, n in [("excluded", f, f + m), ("as female", f + amb, f + m + amb),
                            ("as male", f, f + m + amb)]:
            rows.append({"subject": subj, "ambiguous_labels": amb, "treatment": label,
                         "female": k, "n": n, "reference_female": ref,
                         "p": st.binom_test(k, n, ref)})
    return pd.DataFrame(rows)
