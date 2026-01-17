# stats_active_days_alpha05.py
# Run: python stats_active_days_alpha05.py
#
# Performs hypothesis tests at alpha=0.05 (95% confidence) and prints
# explicit reject/fail-to-reject decisions.

from __future__ import annotations

import math
import numpy as np

try:
    from scipy import stats
except ImportError as e:
    raise SystemExit("This script needs scipy. Install it with: pip install scipy") from e


ALPHA = 0.05  # 95% confidence


def welch_df(s1: float, n1: int, s2: float, n2: int) -> float:
    v1 = (s1**2) / n1
    v2 = (s2**2) / n2
    num = (v1 + v2) ** 2
    den = (v1**2) / (n1 - 1) + (v2**2) / (n2 - 1)
    return num / den


def mean_diff_ci(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int, alpha: float) -> tuple[float, float]:
    diff = m1 - m2
    se = math.sqrt((s1**2) / n1 + (s2**2) / n2)
    df = welch_df(s1, n1, s2, n2)
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    return diff - tcrit * se, diff + tcrit * se


def decision(p: float, alpha: float = ALPHA) -> str:
    return "REJECT H0 (significant)" if p < alpha else "FAIL TO REJECT H0 (not significant)"


def main() -> None:
    # --- Data ---
    counter = np.array([7, 6, 7, 6, 6, 7, 6, 7, 3, 5], dtype=float)
    journal = np.array([4, 3, 3, 1, 6, 2, 4, 4], dtype=float)

    # Descriptives
    m1, m2 = float(counter.mean()), float(journal.mean())
    s1, s2 = float(counter.std(ddof=1)), float(journal.std(ddof=1))
    n1, n2 = counter.size, journal.size

    print(f"Alpha (significance level) = {ALPHA:.2f}  -> confidence level = {1-ALPHA:.2f}\n")
    print(f"Counter: n={n1}, mean={m1:.2f}, sd={s1:.2f}")
    print(f"Journal: n={n2}, mean={m2:.2f}, sd={s2:.2f}\n")

    # Welch t-test
    t_res = stats.ttest_ind(counter, journal, equal_var=False)
    df = welch_df(s1, n1, s2, n2)
    diff = m1 - m2
    ci_lo, ci_hi = mean_diff_ci(m1, s1, n1, m2, s2, n2, alpha=ALPHA)

    print("Welch two-sample t-test")
    print(f"t({df:.2f}) = {t_res.statistic:.2f}, p = {t_res.pvalue:.4f} -> {decision(t_res.pvalue)}")
    print(f"Mean difference (Counter - Journal) = {diff:.2f} days")
    print(f"{int((1-ALPHA)*100)}% CI for mean difference = [{ci_lo:.2f}, {ci_hi:.2f}]\n")

    # Mann–Whitney U test (two-sided)
    try:
        mw = stats.mannwhitneyu(counter, journal, alternative="two-sided", method="exact")
    except TypeError:
        mw = stats.mannwhitneyu(counter, journal, alternative="two-sided")

    print("Mann–Whitney U test (two-sided)")
    print(f"U = {mw.statistic:.0f}, p = {mw.pvalue:.4f} -> {decision(mw.pvalue)}")


if __name__ == "__main__":
    main()
