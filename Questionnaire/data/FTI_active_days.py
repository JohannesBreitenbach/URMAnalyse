# active_days_vs_future_intention.py
# Analysis: Does more engagement (unique active days) relate to higher Future Intention to Use (FTI)?
#
# Data layout (same folder as this script):
# - counter.csv, journal.csv  (questionnaires; delimiter ";"; one row = one participant)
#   columns used:
#     - MOTHER_CODE
#     - the 3 columns starting with "Future Intention to use:"
# - folders: ./data_counter, ./data_journal
#   one CSV per participant; filename = <MOTHER_CODE>.csv
#
# Active days definition:
# - unique calendar dates with >=1 entry
# - multiple entries on the same date count once
#
# FTI definition:
# - mean of the 3 "Future Intention to use:" items (per participant)
#
# Output:
# - console: per-participant table + Pearson r + permutation-test p-value (overall + per condition)
# - out/active_days_vs_fti.csv (merged table)

from __future__ import annotations

import csv
import math
import random
import re
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -------------------------
# Paths
# -------------------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

Q_COUNTER = BASE_DIR / "counter.csv"
Q_JOURNAL = BASE_DIR / "journal.csv"
DIR_COUNTER = BASE_DIR / "data_counter"
DIR_JOURNAL = BASE_DIR / "data_journal"

OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "active_days_vs_fti.csv"

MOTHER_CODE_RE = re.compile(r"^[A-Z]\d{3}[A-Z]$")  # e.g., A016S


# -------------------------
# Helpers
# -------------------------
def norm(s: str) -> str:
    return (s or "").strip().lower()


def detect_delimiter(sample: str) -> str:
    for d in [",", ";", "\t", "|"]:
        if d in sample:
            return d
    return ";"


def read_csv_rows(path: Path, delimiter: str = ";") -> Tuple[List[str], List[List[str]]]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    if not raw:
        return [], []
    r = csv.reader(raw, delimiter=delimiter)
    rows = list(r)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def find_col_idx(header: List[str], exact_name: str) -> int:
    target = norm(exact_name)
    for i, h in enumerate(header):
        if norm(h) == target:
            return i
    raise ValueError(f"Column '{exact_name}' not found in {header}")


def parse_number(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    # handle decimal comma (e.g., 2,666666667)
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


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
]


def try_parse_date(s: str) -> Optional[date]:
    if s is None:
        return None
    txt = str(s).strip()
    if not txt:
        return None

    # ISO-ish first
    try:
        dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
        return dt.date()
    except Exception:
        pass

    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(txt, fmt).date()
        except Exception:
            pass

    return None


def find_participant_file(folder: Path, mother_code: str) -> Optional[Path]:
    p = folder / f"{mother_code}.csv"
    if p.exists():
        return p
    # case-insensitive fallback
    mc = mother_code.strip().lower()
    for f in folder.glob("*.csv"):
        if f.stem.strip().lower() == mc:
            return f
    return None


# -------------------------
# Questionnaire: get FTI per participant
# -------------------------
def read_fti_from_questionnaire(path: Path, condition: str) -> Dict[str, Dict[str, object]]:
    """
    Returns dict: mother_code -> {"condition": condition, "fti": float}
    Uses the 3 columns whose headers start with "Future Intention to use:"
    Skips lines that are not participants.
    """
    header, data = read_csv_rows(path, delimiter=";")
    if not header:
        return {}

    i_code = find_col_idx(header, "MOTHER_CODE")

    # simplest: pick all columns starting with that prefix
    fti_cols = [i for i, h in enumerate(header) if norm(h).startswith("future intention to use:")]
    if len(fti_cols) < 3:
        raise ValueError(
            f"Expected 3 FTI columns starting with 'Future Intention to use:' in {path.name}, "
            f"but found {len(fti_cols)}."
        )
    fti_cols = fti_cols[:3]  # take the first three

    out: Dict[str, Dict[str, object]] = {}
    for row in data:
        if not row or i_code >= len(row):
            continue

        code = row[i_code].strip().upper()
        if not MOTHER_CODE_RE.match(code):
            continue

        vals = []
        ok = True
        for idx in fti_cols:
            if idx >= len(row):
                ok = False
                break
            v = parse_number(row[idx])
            if v is None:
                ok = False
                break
            vals.append(v)

        if not ok:
            continue

        fti = sum(vals) / 3.0
        out[code] = {"condition": condition, "fti": fti}

    return out


# -------------------------
# App data: compute active days
# -------------------------
def unique_dates_counter(app_file: Path) -> List[date]:
    header, data = read_csv_rows(app_file, delimiter=detect_delimiter(app_file.read_text(encoding="utf-8-sig", errors="replace")[:2000]))
    if not header:
        return []

    hn = [norm(h) for h in header]
    dates_set = set()

    # simplest rule: if year/month/day exist -> use them
    if "year" in hn and "month" in hn and "day" in hn:
        iy, im, iday = hn.index("year"), hn.index("month"), hn.index("day")
        for row in data:
            if max(iy, im, iday) >= len(row):
                continue
            try:
                y = int(str(row[iy]).strip())
                m = int(str(row[im]).strip())
                d = int(str(row[iday]).strip())
                dates_set.add(date(y, m, d))
            except Exception:
                continue
        return sorted(dates_set)

    # else: try a "date" column; fallback: first column
    idx = hn.index("date") if "date" in hn else 0
    for row in data:
        if idx >= len(row):
            continue
        d = try_parse_date(row[idx])
        if d is not None:
            dates_set.add(d)
    return sorted(dates_set)


def unique_dates_journal(app_file: Path) -> List[date]:
    # simplest rule (per your setup): date is the second column
    header, data = read_csv_rows(app_file, delimiter=detect_delimiter(app_file.read_text(encoding="utf-8-sig", errors="replace")[:2000]))
    if not header or len(header) < 2:
        return []
    idx = 1

    dates_set = set()
    for row in data:
        if idx >= len(row):
            continue
        d = try_parse_date(row[idx])
        if d is not None:
            dates_set.add(d)
    return sorted(dates_set)


# -------------------------
# Analysis (no scipy): Pearson r + permutation p-value
# -------------------------
def pearson_r(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def permutation_p_value(xs: List[float], ys: List[float], n_perm: int = 10000, seed: int = 42) -> Optional[float]:
    r_obs = pearson_r(xs, ys)
    if r_obs is None:
        return None
    r_obs = abs(r_obs)

    rng = random.Random(seed)
    ys_copy = ys[:]
    count = 0
    for _ in range(n_perm):
        rng.shuffle(ys_copy)
        r_perm = pearson_r(xs, ys_copy)
        if r_perm is None:
            continue
        if abs(r_perm) >= r_obs:
            count += 1

    # add-one smoothing
    return (count + 1) / (n_perm + 1)


def summarize_relation(label: str, xs: List[float], ys: List[float]) -> None:
    r = pearson_r(xs, ys)
    p = permutation_p_value(xs, ys, n_perm=10000)
    n = len(xs)
    if r is None or p is None:
        print(f"{label}: n={n} (insufficient variance/data for correlation)")
        return
    print(f"{label}: n={n}, Pearson r={r:.3f}, permutation p={p:.4f}")


# -------------------------
# Main
# -------------------------
def main() -> None:
    if not Q_COUNTER.exists() or not Q_JOURNAL.exists():
        raise FileNotFoundError("Missing counter.csv or journal.csv next to the script.")
    if not DIR_COUNTER.exists() or not DIR_JOURNAL.exists():
        raise FileNotFoundError("Missing data_counter/ or data_journal/ folders next to the script.")

    meta = {}
    meta.update(read_fti_from_questionnaire(Q_COUNTER, "counter"))
    meta.update(read_fti_from_questionnaire(Q_JOURNAL, "journal"))

    merged_rows = []
    missing_files = []
    skipped_no_fti = 0

    for code in sorted(meta.keys()):
        cond = str(meta[code]["condition"])
        fti = float(meta[code]["fti"])

        folder = DIR_COUNTER if cond == "counter" else DIR_JOURNAL
        app_file = find_participant_file(folder, code)

        if app_file is None:
            missing_files.append(f"{cond}:{code}")
            continue

        if cond == "counter":
            days = len(unique_dates_counter(app_file))
        else:
            days = len(unique_dates_journal(app_file))

        merged_rows.append({"mother_code": code, "condition": cond, "active_days": days, "fti_mean": fti})

    # ---- Console: per-participant table
    print("\nMerged table (active days + FTI mean):")
    for r in merged_rows:
        print(f"- {r['mother_code']}: {r['condition']}, active_days={r['active_days']}, fti_mean={r['fti_mean']:.3f}")

    # ---- Save merged CSV
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mother_code", "condition", "active_days", "fti_mean"])
        w.writeheader()
        for r in merged_rows:
            w.writerow(r)
    print(f"\nSaved: {OUT_CSV.resolve()}")

    # ---- Correlation analysis
    xs_all = [float(r["active_days"]) for r in merged_rows]
    ys_all = [float(r["fti_mean"]) for r in merged_rows]
    print("\nRelationship: active days vs future intention to use (FTI mean of 3 items)")
    summarize_relation("Overall", xs_all, ys_all)

    # per condition
    for cond in ["counter", "journal"]:
        xs = [float(r["active_days"]) for r in merged_rows if r["condition"] == cond]
        ys = [float(r["fti_mean"]) for r in merged_rows if r["condition"] == cond]
        summarize_relation(f"{cond.capitalize()} only", xs, ys)

    if missing_files:
        print("\nMissing app files (excluded):")
        for x in missing_files:
            print(" -", x)


if __name__ == "__main__":
    main()