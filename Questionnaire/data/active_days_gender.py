# active_days_boxplot_4groups.py
# Creates a 4-box boxplot comparing UNIQUE active days per participant:
#   Counter (male), Counter (female), Journal (male), Journal (female)
#
# Data sources (same folder as this script):
# - counter.csv, journal.csv (questionnaire; delimiter ";"; one row = one participant)
#   columns used: GENDER, MOTHER_CODE
# - folders: ./data_counter, ./data_journal
#   one CSV per participant; filename = <MOTHER_CODE>.csv
#
# Active days rule:
# - Active day = a calendar date with >=1 entry
# - Multiple entries on the same date count once
#
# Plot style matches your previous design:
# - Box/whiskers/caps: black outline
# - Median: orange
# - Points: Counter = tab:blue, Journal = tab:orange
#
# Outputs:
# - ./out/active_days_boxplot_4groups.png
# Console:
# - per-participant active days
# - group stats + boxplot stats (q1/median/q3, whiskers)

from __future__ import annotations

import csv
import math
import random
import re
import statistics
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent

Q_COUNTER = BASE_DIR / "counter.csv"
Q_JOURNAL = BASE_DIR / "journal.csv"

DIR_COUNTER = BASE_DIR / "data_counter"
DIR_JOURNAL = BASE_DIR / "data_journal"

OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)
OUT_PNG = OUT_DIR / "active_days_boxplot_4groups.png"

MOTHER_CODE_RE = re.compile(r"^[A-Z]\d{3}[A-Z]$")


# -------------------------
# Questionnaire helpers
# -------------------------
def norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")


def find_col_idx(header: List[str], target: str) -> int:
    t = norm(target)
    hn = [norm(h) for h in header]
    if t in hn:
        return hn.index(t)
    raise ValueError(f"Column '{target}' not found. Found: {header}")


def normalize_gender(g: str) -> str:
    # Assumption from your example files: 1=male, 2=female
    s = (g or "").strip().lower()
    if s in {"1", "m", "male", "man"}:
        return "male"
    if s in {"2", "f", "female", "woman"}:
        return "female"
    return s or "unknown"


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
            if not row or max(i_gender, i_code) >= len(row):
                continue

            code = row[i_code].strip().upper()
            if not MOTHER_CODE_RE.match(code):
                continue

            gender = normalize_gender(row[i_gender])
            participants[code] = {"condition": condition, "gender": gender}

    return participants


# -------------------------
# App CSV reading + date parsing
# -------------------------
_DT_FORMATS = [
    "%Y-%m-%d",
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
]


def detect_delimiter(sample: str) -> str:
    for d in [",", ";", "\t", "|"]:
        if d in sample:
            return d
    return ","


def read_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    if not raw:
        return [], []
    delim = detect_delimiter("\n".join(raw[:50]))
    rows = list(csv.reader(raw, delimiter=delim))
    if not rows:
        return [], []
    header = rows[0]
    data = rows[1:] if len(rows) > 1 else []
    return header, data


def try_parse_datetime(s: str) -> Optional[datetime]:
    if s is None:
        return None
    txt = str(s).strip()
    if not txt:
        return None

    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(txt, fmt)
        except Exception:
            pass

    # ISO-ish fallback (handles many "YYYY-MM-DDTHH:MM:SS.sss" without tz)
    try:
        return datetime.fromisoformat(txt.replace("Z", "+00:00"))
    except Exception:
        return None


def find_col(header: List[str], candidates: List[str]) -> Optional[int]:
    hn = [norm(h) for h in header]

    # exact match
    for cand in candidates:
        c = norm(cand)
        if c in hn:
            return hn.index(c)

    # substring match
    for cand in candidates:
        c = norm(cand)
        for i, h in enumerate(hn):
            if c in h:
                return i

    return None


def unique_dates_from_counter_file(path: Path) -> List[date]:
    header, data = read_rows(path)
    if not header:
        return []

    hn = [norm(h) for h in header]

    # simplest: year/month/day columns if present
    if all(k in hn for k in ["year", "month", "day"]):
        iy, im, iday = hn.index("year"), hn.index("month"), hn.index("day")
        out = set()
        for row in data:
            if max(iy, im, iday) >= len(row):
                continue
            try:
                y = int(row[iy])
                m = int(row[im])
                d = int(row[iday])
                out.add(date(y, m, d))
            except Exception:
                continue
        return sorted(out)

    # otherwise: one date/timestamp column
    idx = find_col(header, ["date", "timestamp", "datetime", "created_at", "time", "created", "ts"])
    if idx is None:
        idx = 0  # fallback: first column

    out = set()
    for row in data:
        if idx >= len(row):
            continue
        dt = try_parse_datetime(row[idx])
        if dt is not None:
            out.add(dt.date())
    return sorted(out)


def unique_dates_from_journal_file(path: Path) -> List[date]:
    header, data = read_rows(path)
    if not header:
        return []
    # per your earlier rule: date is 2nd column
    if len(header) < 2:
        return []

    idx = 1
    out = set()
    for row in data:
        if idx >= len(row):
            continue
        dt = try_parse_datetime(row[idx])
        if dt is not None:
            out.add(dt.date())
    return sorted(out)


# -------------------------
# Stats and boxplot numbers (Tukey)
# -------------------------
def percentile(sorted_vals: List[float], p: float) -> float:
    """Linear interpolation percentile (0..100) on sorted values."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])

    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def tukey_box_stats(vals: List[int]) -> Dict[str, float]:
    """
    Returns q1, median, q3, whisker_low, whisker_high (Tukey 1.5*IQR),
    plus min/max (actual).
    """
    if not vals:
        return dict(q1=0, median=0, q3=0, wl=0, wh=0, vmin=0, vmax=0)

    s = sorted(float(v) for v in vals)
    q1 = percentile(s, 25)
    med = percentile(s, 50)
    q3 = percentile(s, 75)
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr

    wl = min(v for v in s if v >= low_fence)
    wh = max(v for v in s if v <= high_fence)

    return dict(q1=q1, median=med, q3=q3, wl=wl, wh=wh, vmin=min(s), vmax=max(s))


# -------------------------
# Plot
# -------------------------
def make_boxplot_4groups_active_days(groups: Dict[str, List[int]], out_path: Path) -> None:
    order = ["counter_male", "counter_female", "journal_male", "journal_female"]
    labels = ["Counter (male)", "Counter (female)", "Journal (male)", "Journal (female)"]
    positions = [1, 2, 3, 4]
    data = [groups.get(k, []) for k in order]

    c_counter = "tab:blue"
    c_journal = "tab:orange"
    point_colors = [c_counter, c_counter, c_journal, c_journal]

    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)

    ax.boxplot(
        data,
        positions=positions,
        widths=0.3,
        showfliers=False,
        patch_artist=False,
        boxprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=2),
        capprops=dict(color="black", linewidth=2),
        medianprops=dict(color=c_journal, linewidth=2),
    )

    # Points aligned (no jitter) -> exactly centered on each box
    for x, vals, col in zip(positions, data, point_colors):
        ax.scatter([x] * len(vals), vals, s=150, color=col, alpha=0.85, zorder=3)

    ax.set_title("Engagement by condition (unique active days)", fontsize=26, pad=18)
    ax.set_ylabel("Active days per participant", fontsize=22)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=20, rotation=15, ha="right")
    ax.tick_params(axis="y", labelsize=18)

    ax.grid(axis="y", linestyle="--", linewidth=1.2, alpha=0.45)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.88)
    fig.savefig(out_path)
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main() -> None:
    if not Q_COUNTER.exists() or not Q_JOURNAL.exists():
        raise FileNotFoundError("Missing counter.csv or journal.csv next to the script.")
    if not DIR_COUNTER.exists() or not DIR_JOURNAL.exists():
        raise FileNotFoundError("Missing data_counter/ or data_journal/ folders next to the script.")

    participants = {}
    participants.update(read_questionnaire(Q_COUNTER, "counter"))
    participants.update(read_questionnaire(Q_JOURNAL, "journal"))

    # collect per participant
    per_participant = []  # (condition, gender, mother_code, active_days)
    missing_files = []

    for code in sorted(participants.keys()):
        cond = participants[code]["condition"]
        gender = participants[code]["gender"]

        folder = DIR_COUNTER if cond == "counter" else DIR_JOURNAL
        app_path = folder / f"{code}.csv"

        if not app_path.exists():
            missing_files.append(f"{cond}:{code}")
            continue

        if cond == "counter":
            uniq_dates = unique_dates_from_counter_file(app_path)
        else:
            uniq_dates = unique_dates_from_journal_file(app_path)

        active_days = len(uniq_dates)
        per_participant.append((cond, gender, code, active_days))

    # build 4 groups (male/female only, like your other plot)
    groups: Dict[str, List[int]] = {
        "counter_male": [],
        "counter_female": [],
        "journal_male": [],
        "journal_female": [],
    }
    excluded_gender = []

    for cond, gender, code, days in per_participant:
        if gender not in {"male", "female"}:
            excluded_gender.append(f"{cond}:{code} ({gender})")
            continue
        groups[f"{cond}_{gender}"].append(days)

    # ---- console output: displayed values
    print("\nActive days per participant (unique dates):")
    for cond, gender, code, days in per_participant:
        print(f"- {code}: {cond}, {gender}, active_days={days}")

    print("\nGroup stats (the values shown by the plot, plus Tukey box stats):")
    for key in ["counter_male", "counter_female", "journal_male", "journal_female"]:
        vals = groups[key]
        n = len(vals)
        m = sum(vals) / n if n else 0.0
        sd = statistics.stdev(vals) if n >= 2 else 0.0
        b = tukey_box_stats(vals)
        print(
            f"- {key}: n={n}, M={m:.2f}, SD={sd:.2f}, "
            f"q1={b['q1']:.2f}, median={b['median']:.2f}, q3={b['q3']:.2f}, "
            f"whiskers=[{b['wl']:.2f}, {b['wh']:.2f}], min={b['vmin']:.2f}, max={b['vmax']:.2f}, "
            f"values={sorted(vals)}"
        )

    if excluded_gender:
        print("\nExcluded (gender not male/female):")
        for x in excluded_gender:
            print(" -", x)

    if missing_files:
        print("\nMissing app files (excluded):")
        for x in missing_files:
            print(" -", x)

    # plot
    make_boxplot_4groups_active_days(groups, OUT_PNG)
    print(f"\nSaved plot: {OUT_PNG}")


if __name__ == "__main__":
    main()