"""Statistical tests, effect sizes and resampling used by the audit.

Samples are tiny (5 images per model/prompt cell, 15 per prompt), so every test is
exact or Monte Carlo exact rather than relying on chi-square asymptotics.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

N_BOOT = 10_000
N_PERM = 100_000


# ---------- single proportion vs a reference ----------

def binom_test(k: int, n: int, p0: float) -> float:
    """Two-sided exact binomial test (minimum-likelihood method)."""
    return stats.binomtest(k, n, p0).pvalue


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=level, method="exact")
    return ci.low, ci.high


def cohen_h(p: float, p0: float) -> float:
    return 2 * np.arcsin(np.sqrt(p)) - 2 * np.arcsin(np.sqrt(p0))


def min_attainable_p(n: int, p0: float) -> float:
    """Smallest two-sided p-value any outcome could produce: a power diagnostic."""
    return min(binom_test(k, n, p0) for k in range(n + 1))


def bootstrap_share(successes: np.ndarray, totals: np.ndarray, rng: np.random.Generator,
                    n_boot: int = N_BOOT, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI for a pooled share, resampling images within each model."""
    successes, totals = np.asarray(successes), np.asarray(totals)
    draws = rng.binomial(totals, successes / np.where(totals == 0, 1, totals),
                         size=(n_boot, len(totals)))
    pooled = draws.sum(1) / totals.sum()
    a = (1 - level) / 2
    return tuple(np.quantile(pooled, [a, 1 - a]))


def bootstrap_gap(successes: np.ndarray, totals: np.ndarray, rng: np.random.Generator,
                  n_boot: int = N_BOOT, level: float = 0.95) -> tuple[float, float, float]:
    """Max-min share across models (demographic parity difference) with bootstrap CI."""
    successes, totals = np.asarray(successes), np.asarray(totals)
    rates = successes / totals
    draws = rng.binomial(totals, rates, size=(n_boot, len(totals))) / totals
    gaps = draws.max(1) - draws.min(1)
    a = (1 - level) / 2
    lo, hi = np.quantile(gaps, [a, 1 - a])
    return rates.max() - rates.min(), lo, hi


# ---------- contingency tables ----------

def _chi2_stat(obs: np.ndarray) -> np.ndarray:
    """Pearson chi-square for a stack of tables (..., r, c); zero-margin cells contribute 0."""
    obs = obs.astype(float)
    n = obs.sum((-2, -1), keepdims=True)
    exp = obs.sum(-1, keepdims=True) * obs.sum(-2, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(exp > 0, (obs - exp) ** 2 / exp, 0.0)
    return terms.sum((-2, -1))


def drop_empty(table: np.ndarray) -> np.ndarray:
    table = np.asarray(table)
    return table[table.sum(1) > 0][:, table.sum(0) > 0]


def permutation_independence(table: np.ndarray, rng: np.random.Generator,
                             n_perm: int = N_PERM) -> tuple[float, float]:
    """Monte Carlo exact (Freeman-Halton style) test of independence for an r x c table.

    Margins are held fixed by permuting column labels across observations.
    Returns (chi2 statistic, p-value).
    """
    table = drop_empty(table)
    r, c = table.shape
    if r < 2 or c < 2:
        return 0.0, 1.0
    rows = np.repeat(np.arange(r), table.sum(1))
    cols = np.repeat(np.tile(np.arange(c), r), table.ravel())
    observed = _chi2_stat(table)
    hits, done, batch = 0, 0, 20_000
    while done < n_perm:
        b = min(batch, n_perm - done)
        perm = rng.permuted(np.broadcast_to(cols, (b, cols.size)), axis=1)
        idx = rows * c + perm
        tabs = np.zeros((b, r * c), dtype=int)
        np.add.at(tabs, (np.arange(b)[:, None], idx), 1)
        hits += int((_chi2_stat(tabs.reshape(b, r, c)) >= observed - 1e-9).sum())
        done += b
    return float(observed), (hits + 1) / (n_perm + 1)


def cramers_v(table: np.ndarray) -> float:
    table = drop_empty(table)
    if min(table.shape) < 2:
        return 0.0
    return float(np.sqrt(_chi2_stat(table) / (table.sum() * (min(table.shape) - 1))))


def bootstrap_cramers_v(table: np.ndarray, rng: np.random.Generator,
                        n_boot: int = N_BOOT, level: float = 0.95) -> tuple[float, float]:
    """Percentile CI for Cramer's V, resampling images within each row (model/prompt)."""
    table = np.asarray(table)
    probs = table / table.sum(1, keepdims=True)
    vs = np.empty(n_boot)
    for i in range(n_boot):
        boot = np.vstack([rng.multinomial(n, p) for n, p in zip(table.sum(1), probs)])
        vs[i] = cramers_v(boot)
    a = (1 - level) / 2
    return tuple(np.quantile(vs, [a, 1 - a]))


def multinomial_gof(counts: np.ndarray, probs: np.ndarray, rng: np.random.Generator,
                    n_sim: int = N_PERM) -> tuple[float, float]:
    """Monte Carlo exact chi-square goodness-of-fit test against reference probabilities."""
    counts, probs = np.asarray(counts), np.asarray(probs, dtype=float)
    n = counts.sum()
    exp = n * probs
    observed = float(((counts - exp) ** 2 / exp).sum())
    sims = rng.multinomial(n, probs, size=n_sim)
    sim_stats = ((sims - exp) ** 2 / exp).sum(1)
    return observed, (int((sim_stats >= observed - 1e-9).sum()) + 1) / (n_sim + 1)


def cohen_w(counts: np.ndarray, probs: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    p_obs = counts / counts.sum()
    return float(np.sqrt(((p_obs - probs) ** 2 / probs).sum()))


# ---------- multiplicity ----------

def adjust(pvals, method: str) -> np.ndarray:
    return multipletests(np.asarray(pvals), method=method)[1]


# ---------- robustness ----------

def fragility(k: int, n: int, p0: float, alpha: float) -> int:
    """Label flips needed to reverse the significance decision of a binomial test.

    For a significant result: flips toward p0 until p >= alpha (classic fragility index).
    For a non-significant result: flips away from p0 until p < alpha (reverse fragility).
    Returns -1 if no number of flips changes the decision.
    """
    significant = binom_test(k, n, p0) < alpha
    # A significant result is reversed by moving toward p0; a non-significant one may
    # become significant in either direction, so search both and keep the nearest.
    if significant:
        steps = [1 if k / n < p0 else -1]
    else:
        steps = [1, -1]
    best = -1
    for step in steps:
        for flips in range(1, n + 1):
            k2 = k + step * flips
            if not 0 <= k2 <= n:
                break
            if (binom_test(k2, n, p0) < alpha) != significant:
                best = flips if best < 0 else min(best, flips)
                break
    return best
