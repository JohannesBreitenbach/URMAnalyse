# stats_active_days.py
# Run: python stats_active_days.py
#
# Computes descriptive stats + Welch t-test + Mann–Whitney U test +
# Hedges' g + common-language effect size, then prints a report.

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np

try:
    from scipy import stats
except ImportError as e:
    raise SystemExit(
        "This script needs scipy. Install it with: pip install scipy"
    ) from e


def describe(x: Iterable[float]) -> dict:
    a = np.asarray(list(x), dtype=float)
    n = a.size
    mean = float(np.mean(a))
    median = float(np.median(a))
    sd = float(np.std(a, ddof=1)) if n > 1 else float("nan")
    mn = float(np.min(a))
    q1 = float(np.quantile(a, 0.25, method="linear"))
    q3 = float(np.quantile(a, 0.75, method="linear"))
    iqr = q3 - q1
    mx = float(np.max(a))
    return {
        "n": n,
        "values": a,
        "mean": mean,
        "median": median,
        "sd": sd,
        "min": mn,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "max": mx,
    }


def welch_df(s1: float, n1: int, s2: float, n2: int) -> float:
    # Welch–Satterthwaite degrees of freedom
    v1 = (s1**2) / n1
    v2 = (s2**2) / n2
    num = (v1 + v2) ** 2
    den = (v1**2) / (n1 - 1) + (v2**2) / (n2 - 1)
    return num / den


def mean_diff_ci(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int, alpha: float = 0.05) -> Tuple[float, float]:
    diff = m1 - m2
    se = math.sqrt((s1**2) / n1 + (s2**2) / n2)
    df = welch_df(s1, n1, s2, n2)
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    lo = diff - tcrit * se
    hi = diff + tcrit * se
    return lo, hi


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    n1, n2 = len(x), len(y)
    s1 = np.std(x, ddof=1)
    s2 = np.std(y, ddof=1)

    # pooled SD
    sp = math.sqrt(((n1 - 1) * (s1**2) + (n2 - 1) * (s2**2)) / (n1 + n2 - 2))
    if sp == 0:
        return float("nan")

    d = (np.mean(x) - np.mean(y)) / sp
    df = n1 + n2 - 2
    J = 1 - (3 / (4 * df - 1))  # small-sample correction
    return float(J * d)


def common_language_effect_size(x: np.ndarray, y: np.ndarray) -> float:
    # P(X > Y) with ties counted as 0.5
    wins = 0.0
    total = len(x) * len(y)
    for xi in x:
        for yj in y:
            if xi > yj:
                wins += 1.0
            elif xi == yj:
                wins += 0.5
    return float(wins / total)


def print_group(name: str, desc: dict) -> None:
    vals = ", ".join(str(int(v)) if float(v).is_integer() else str(v) for v in desc["values"])
    print(f"{name} (active days per participant)")
    print(f"n = {desc['n']}")
    print(f"values = {vals}")
    print(f"mean = {desc['mean']:.2f}")
    print(f"median = {desc['median']:.2f}")
    print(f"sd = {desc['sd']:.2f}")
    print(f"min = {desc['min']:.2f}")
    print(f"q1 (25%) = {desc['q1']:.2f}")
    print(f"q3 (75%) = {desc['q3']:.2f}")
    print(f"iqr = {desc['iqr']:.2f}")
    print(f"max = {desc['max']:.2f}")
    print()


def main() -> None:
    # --- Your data here ---
    counter = np.array([7, 6, 7, 6, 6, 7, 6, 7, 3, 5], dtype=float)
    journal = np.array([4, 3, 3, 1, 6, 2, 4, 4], dtype=float)

    # Descriptives
    d_counter = describe(counter)
    d_journal = describe(journal)

    print_group("Counter group", d_counter)
    print_group("Journal group", d_journal)

    # Welch t-test
    t_res = stats.ttest_ind(counter, journal, equal_var=False)
    s1, s2 = d_counter["sd"], d_journal["sd"]
    n1, n2 = d_counter["n"], d_journal["n"]
    df = welch_df(s1, n1, s2, n2)

    diff = d_counter["mean"] - d_journal["mean"]
    ci_lo, ci_hi = mean_diff_ci(d_counter["mean"], s1, n1, d_journal["mean"], s2, n2)

    print("Welch two-sample t-test")
    print(f"mean difference (Counter - Journal) = {diff:.2f} days")
    print(f"t({df:.2f}) = {t_res.statistic:.2f}, p = {t_res.pvalue:.4f}")
    print(f"95% CI for mean difference = [{ci_lo:.2f}, {ci_hi:.2f}]")
    print()

    # Mann–Whitney U (two-sided). Try exact if available.
    try:
        mw = stats.mannwhitneyu(counter, journal, alternative="two-sided", method="exact")
    except TypeError:
        mw = stats.mannwhitneyu(counter, journal, alternative="two-sided")
    print("Mann–Whitney U test (two-sided)")
    print(f"U = {mw.statistic:.0f}, p = {mw.pvalue:.4f}")
    print()

    # Effect sizes
    g = hedges_g(counter, journal)
    cles = common_language_effect_size(counter, journal)

    print("Effect sizes")
    print(f"Hedges' g = {g:.2f}")
    print(f"Common-language effect size P(Counter > Journal) = {cles:.2f}")
    print()


if __name__ == "__main__":
    main()
