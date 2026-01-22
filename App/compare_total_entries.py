# compare_total_entries_simple_no_pandas.py
# Compares all CSVs in ./data_counter and ./data_journal.
# Question answered: "What is the average total number of entries per participant per condition?"
# Rules:
# - One file = one participant
# - Total entries per participant = (number of rows in file) - 1  (header row)
# Output includes CHI-friendly descriptives: n, M, SD, SEM, 95% CI, median, min, max.
# - No pandas

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
COUNTER_DIR = BASE_DIR / "data_counter"
JOURNAL_DIR = BASE_DIR / "data_journal"
OUT_CSV = BASE_DIR / "total_entries_compare.csv"


def count_rows_minus_header(path: Path) -> int:
    """
    Counts total rows using csv.reader, subtracts 1 for the header.
    Returns 0 for empty files or files with only a header.
    """
    seps = [",", ";", "\t", "|"]

    raw = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    if not raw:
        return 0

    best_rows = None  # (n_cols_header, rows)
    for sep in seps:
        try:
            rows = list(csv.reader(raw, delimiter=sep))
            if not rows:
                continue
            header_cols = len(rows[0])
            if best_rows is None or header_cols > best_rows[0]:
                best_rows = (header_cols, rows)
        except Exception:
            continue

    if best_rows is None:
        return 0

    rows = best_rows[1]
    return max(len(rows) - 1, 0)


def folder_totals(folder: Path) -> list[int]:
    if not folder.exists():
        return []
    totals: list[int] = []
    for p in sorted(folder.glob("*.csv")):
        totals.append(count_rows_minus_header(p))
    return totals


def t_crit_95_two_tailed(df: int) -> float:
    """
    Two-tailed 95% t critical value (alpha=0.05) for df=1..30 (common CHI sample sizes).
    Falls back to normal approx (1.96) outside table.
    Source values are standard t-table constants.
    """
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return table.get(df, 1.96)


def descriptives(values: list[int]) -> dict:
    n = len(values)
    if n == 0:
        return {
            "n": 0, "mean": 0.0, "sd": 0.0, "sem": 0.0, "ci95": 0.0,
            "median": 0.0, "min": 0, "max": 0
        }

    mean_v = sum(values) / n
    sd_v = statistics.stdev(values) if n >= 2 else 0.0
    sem_v = sd_v / math.sqrt(n) if n >= 1 else 0.0
    tcrit = t_crit_95_two_tailed(n - 1) if n >= 2 else 0.0
    ci95_v = tcrit * sem_v if n >= 2 else 0.0

    return {
        "n": n,
        "mean": mean_v,
        "sd": sd_v,
        "sem": sem_v,
        "ci95": ci95_v,
        "median": float(statistics.median(values)),
        "min": min(values),
        "max": max(values),
    }


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "condition",
        "n",
        "mean_total_entries",
        "sd_total_entries",
        "sem",
        "ci95_halfwidth",
        "median",
        "min",
        "max",
        "sum_total_entries",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    counter_vals = folder_totals(COUNTER_DIR)
    journal_vals = folder_totals(JOURNAL_DIR)

    counter_desc = descriptives(counter_vals)
    journal_desc = descriptives(journal_vals)

    def pack(name: str, vals: list[int], d: dict) -> dict:
        return {
            "condition": name,
            "n": d["n"],
            "mean_total_entries": f"{d['mean']:.6f}",
            "sd_total_entries": f"{d['sd']:.6f}",
            "sem": f"{d['sem']:.6f}",
            "ci95_halfwidth": f"{d['ci95']:.6f}",
            "median": f"{d['median']:.6f}",
            "min": d["min"],
            "max": d["max"],
            "sum_total_entries": sum(vals),
        }

    summary_rows = [
        pack("counter", counter_vals, counter_desc),
        pack("journal", journal_vals, journal_desc),
    ]

    # Console output (CHI-friendly)
    print("Total entries per condition (rowcount - 1 per participant file):\n")
    for name, d in [("Counter", counter_desc), ("Journal", journal_desc)]:
        if d["n"] == 0:
            print(f"{name}: n=0")
            continue
        # M (SD), plus SEM and 95% CI half-width
        print(
            f"{name}: n={d['n']}, M={d['mean']:.2f}, SD={d['sd']:.2f}, "
            f"SEM={d['sem']:.2f}, 95% CI ±{d['ci95']:.2f}, "
            f"median={d['median']:.2f}, min={d['min']}, max={d['max']}"
        )

    write_summary_csv(OUT_CSV, summary_rows)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
