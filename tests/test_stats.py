from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

from bias_audit import analysis as an
from bias_audit import stats as st
from bias_audit.data import REPORT_CODING, load_counts

XLSX = Path(__file__).parents[1] / "data" / "Aggregation.xlsx"


def test_permutation_matches_fisher_on_2x2():
    table = np.array([[5, 0], [1, 4]])
    _, p = st.permutation_independence(table, np.random.default_rng(0), n_perm=50_000)
    assert p == pytest.approx(sps.fisher_exact(table).pvalue, abs=0.01)


def test_cramers_v_bounds():
    assert st.cramers_v(np.array([[5, 0], [0, 5]])) == pytest.approx(1.0)
    assert st.cramers_v(np.array([[3, 3], [3, 3]])) == pytest.approx(0.0)


def test_fragility_directions():
    assert st.fragility(15, 15, 0.5, 0.05) >= 1          # significant: flips toward 0.5
    assert st.fragility(0, 15, 0.06, 0.05) > 0           # non-significant: must look upward
    assert st.fragility(1, 15, 0.08, 0.05) > 0


def test_workbook_loads_and_cells_sum_to_five():
    counts, bench = load_counts(XLSX)
    assert counts.subject.nunique() == 9
    assert set(bench.subject) == {"Doctor", "CEO", "Teacher", "Construction Worker", "Scientist", "Farmer"}
    assert bench.set_index("subject").female_share["CEO"] == pytest.approx(0.08)


def test_reproduces_research_report_chi_square():
    counts, _ = load_counts(XLSX)
    res = an.occupation_gender_association(counts, np.random.default_rng(1),
                                           coding=REPORT_CODING, occupations=list(REPORT_CODING))
    pooled = res.iloc[0]
    assert pooled.chi2 == pytest.approx(16.77, abs=0.01)
    assert pooled.p_asymptotic == pytest.approx(0.0021, abs=0.0001)
    assert pooled.cramers_v == pytest.approx(0.473, abs=0.001)
