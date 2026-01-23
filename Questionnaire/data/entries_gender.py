# entries_by_gender_and_condition.py
# Simplest version for your file formats + a 4-box boxplot.
#
# Inputs (same folder as this script):
# - counter.csv, journal.csv  (questionnaire; delimiter ";"; one row = one participant)
#   columns used: GENDER, MOTHER_CODE
# - folders: ./data_counter, ./data_journal
#   one file per participant; filename = <MOTHER_CODE>.csv
#
# App entries counting rule:
# - entries = (number of non-empty lines in the app file) - 1  (header)
#
# Outputs:
# - entries_per_participant.csv  (mother_code, condition, gender, entries, missing_app_file)
# - entries_summary.csv          (means per condition, per gender, and condition×gender)
# - entries_boxplot_4groups.png  (4 boxes: Counter×male, Counter×female, Journal×male, Journal×female)
#
# No pandas.

from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import random
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent

Q_COUNTER = BASE_DIR / "counter.csv"
Q_JOURNAL = BASE_DIR / "journal.csv"

DIR_COUNTER = BASE_DIR / "data_counter"
DIR_JOURNAL = BASE_DIR / "data_journal"

OUT_PARTICIPANTS = BASE_DIR / "entries_per_participant.csv"
OUT_SUMMARY = BASE_DIR / "entries_summary.csv"
OUT_BOXPLOT = BASE_DIR / "entries_boxplot_4groups.png"

# Your mother codes look like A016S, J013A, ...
MOTHER_CODE_RE = re.compile(r"^[A-Z]\d{3}[A-Z]$")


def norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")


def find_col_idx(header: List[str], target: str) -> int:
    """
    Case-insensitive match for column names.
    Accepts minor variations like spaces vs underscores.
    """
    t = norm(target)
    hn = [norm(h) for h in header]
    if t in hn:
        return hn.index(t)
    raise ValueError(f"Column '{target}' not found. Found: {header}")


def normalize_gender(g: str) -> str:
    """
    Questionnaire uses numeric gender codes (seen: 1, 2).
    Assumption: 1=male, 2=female. If yours is reversed, swap here.
    """
    s = (g or "").strip().lower()
    if s in {"1", "m", "male", "man"}:
        return "male"
    if s in {"2", "f", "female", "woman"}:
        return "female"
    if not s:
        return "unknown"
    return s


def read_questionnaire(path: Path, condition: str) -> Dict[str, Dict[str, str]]:
    """
    Returns dict: mother_code -> {"condition": ..., "gender": ...}
    Skips non-participant lines by validating mother code pattern.
    """
    participants: Dict[str, Dict[str, str]] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f, delimiter=";")
        header = next(r, None)
        if not header:
            return participants

        i_gender = find_col_idx(header, "GENDER")
        i_code = find_col_idx(header, "MOTHER_CODE")

        for row in r:
            if not row:
                continue
            if max(i_gender, i_code) >= len(row):
                continue

            code = row[i_code].strip().upper()
            if not MOTHER_CODE_RE.match(code):
                continue  # ignore summary/garbage lines

            gender = normalize_gender(row[i_gender])
            participants[code] = {"condition": condition, "gender": gender}

    return participants


def count_entries(app_file: Path) -> Optional[int]:
    """entries = non-empty lines - 1 (header). Returns None if file missing."""
    if not app_file.exists():
        return None
    lines = app_file.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    non_empty = sum(1 for ln in lines if ln.strip())
    return max(non_empty - 1, 0)


def stats(values: List[int]) -> Tuple[int, float, float]:
    """Returns (n, mean, sd)."""
    n = len(values)
    if n == 0:
        return 0, 0.0, 0.0
    mean = sum(values) / n
    sd = statistics.stdev(values) if n >= 2 else 0.0
    return n, mean, sd


def make_boxplot_4groups(groups: dict[str, list[int]], out_path):
    order = ["counter_male", "counter_female", "journal_male", "journal_female"]
    labels = ["Counter (male)", "Counter (female)", "Journal (male)", "Journal (female)"]
    positions = [1, 2, 3, 4]
    data = [groups.get(k, []) for k in order]

    c_counter = "tab:blue"
    c_journal = "tab:orange"
    point_colors = [c_counter, c_counter, c_journal, c_journal]

    # Bigger canvas so the “exact style” fonts don’t collide
    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)

    ax.boxplot(
        data,
        positions=positions,
        widths=0.3,
        showfliers=False,
        patch_artist=False,  # outlines only
        boxprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=2),
        capprops=dict(color="black", linewidth=2),
        medianprops=dict(color=c_journal, linewidth=2),  # orange median line
    )

    # Overlay points WITHOUT jitter (aligned)
    for x, vals, col in zip(positions, data, point_colors):
        ax.scatter([x] * len(vals), vals, s=150, color=col, alpha=0.85, zorder=3)


    # Keep the same “design language” but prevent overlap
    ax.set_title("Entries by Gender", fontsize=26, pad=18)
    ax.set_ylabel("Total Entry Count", fontsize=22)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=20, rotation=15, ha="right")
    ax.tick_params(axis="y", labelsize=18)

    ax.grid(axis="y", linestyle="--", linewidth=1.2, alpha=0.45)

    # Manual margins to avoid cutting title/labels
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.88)

    fig.savefig(out_path)
    plt.close(fig)

def main() -> None:
    if not Q_COUNTER.exists() or not Q_JOURNAL.exists():
        raise FileNotFoundError("Missing counter.csv or journal.csv next to the script.")

    if not DIR_COUNTER.exists() or not DIR_JOURNAL.exists():
        raise FileNotFoundError("Missing data_counter/ or data_journal/ folders next to the script.")

    # 1) read questionnaire participant metadata
    participants = {}
    participants.update(read_questionnaire(Q_COUNTER, "counter"))
    participants.update(read_questionnaire(Q_JOURNAL, "journal"))

    # 2) connect to app files and build per-participant rows
    rows = []
    missing = []

    for code in sorted(participants.keys()):
        cond = participants[code]["condition"]
        gender = participants[code]["gender"]

        folder = DIR_COUNTER if cond == "counter" else DIR_JOURNAL
        app_path = folder / f"{code}.csv"

        entries = count_entries(app_path)
        missing_flag = 0 if entries is not None else 1
        if entries is None:
            entries = 0
            missing.append(f"{cond}:{code}")

        rows.append(
            {
                "mother_code": code,
                "condition": cond,
                "gender": gender,
                "entries": entries,
                "missing_app_file": missing_flag,
            }
        )

    # 3) write participant table
    with OUT_PARTICIPANTS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["mother_code", "condition", "gender", "entries", "missing_app_file"],
        )
        w.writeheader()
        w.writerows(rows)

    # 4) summaries (exclude missing_app_file=1 from means)
    by_condition: Dict[str, List[int]] = {}
    by_gender: Dict[str, List[int]] = {}
    by_cond_gender: Dict[str, List[int]] = {}

    # for 4-group plot
    plot_groups: Dict[str, List[int]] = {
        "counter_male": [],
        "counter_female": [],
        "journal_male": [],
        "journal_female": [],
    }
    excluded_for_plot = []  # e.g., unknown/diverse genders

    for r in rows:
        if int(r["missing_app_file"]) == 1:
            continue

        cond = str(r["condition"])
        gender = str(r["gender"])
        e = int(r["entries"])

        by_condition.setdefault(cond, []).append(e)
        by_gender.setdefault(gender, []).append(e)
        by_cond_gender.setdefault(f"{cond}×{gender}", []).append(e)

        if gender in {"male", "female"}:
            plot_groups[f"{cond}_{gender}"].append(e)
        else:
            excluded_for_plot.append(f"{cond}:{r['mother_code']} ({gender})")

    # 5) write summary csv
    with OUT_SUMMARY.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_type", "group", "n", "mean_entries", "sd_entries"])

        for cond in sorted(by_condition):
            n, m, sd = stats(by_condition[cond])
            w.writerow(["condition", cond, n, f"{m:.6f}", f"{sd:.6f}"])

        for g in sorted(by_gender):
            n, m, sd = stats(by_gender[g])
            w.writerow(["gender", g, n, f"{m:.6f}", f"{sd:.6f}"])

        for cg in sorted(by_cond_gender):
            n, m, sd = stats(by_cond_gender[cg])
            w.writerow(["condition_x_gender", cg, n, f"{m:.6f}", f"{sd:.6f}"])

    # 6) boxplot (4 boxes)
    make_boxplot_4groups(plot_groups, OUT_BOXPLOT)

    # 7) console output
    print(f"Saved: {OUT_PARTICIPANTS.name}")
    print(f"Saved: {OUT_SUMMARY.name}")
    print(f"Saved: {OUT_BOXPLOT.name}")

    print("\nBoxplot group sizes:")
    for k in ["counter_male", "counter_female", "journal_male", "journal_female"]:
        print(f"- {k}: n={len(plot_groups[k])}")

    if excluded_for_plot:
        print("\nExcluded from 4-box plot (gender not male/female):")
        for x in excluded_for_plot:
            print(" -", x)

    if missing:
        print("\nMissing app files (excluded from summaries and plot):")
        for x in missing:
            print(" -", x)


if __name__ == "__main__":
    main()