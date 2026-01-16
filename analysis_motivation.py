# compare_daily_participants.py
# Compares all CSVs in ./data_counter and ./data_journal.
# Question answered: "How many participants used the app on a given date?"
# Rules:
# - One file = one participant
# - Multiple entries on the same date for the same participant count only once

from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
COUNTER_DIR = BASE_DIR / "data_counter"
JOURNAL_DIR = BASE_DIR / "data_journal"
OUT_CSV = BASE_DIR / "daily_participants_compare.csv"


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Try to read CSV with common separators. Falls back to pandas auto/sniffing."""
    for sep in [",", ";", "\t", "|"]:
        try:
            return pd.read_csv(path, sep=sep)
        except Exception:
            pass
    # last resort: sniff
    return pd.read_csv(path, sep=None, engine="python")


def extract_dates_counter(df: pd.DataFrame) -> pd.Series:
    cols = {c.strip().lower(): c for c in df.columns}

    if {"year", "month", "day"}.issubset(cols.keys()):
        dt = pd.to_datetime(
            dict(
                year=pd.to_numeric(df[cols["year"]], errors="coerce"),
                month=pd.to_numeric(df[cols["month"]], errors="coerce"),
                day=pd.to_numeric(df[cols["day"]], errors="coerce"),
            ),
            errors="coerce",
        )
        return dt

    # if already has a date-like column
    if "date" in cols:
        return pd.to_datetime(df[cols["date"]], errors="coerce", dayfirst=True)

    # fallback: try first column as date (rare, but happens)
    return pd.to_datetime(df.iloc[:, 0], errors="coerce", dayfirst=True)


def extract_dates_journal(df: pd.DataFrame) -> pd.Series:
    # Requirement: keep only first two columns (id, date) -> date is second column
    if df.shape[1] < 2:
        raise ValueError("Journal CSV has < 2 columns; cannot read date from 2nd column.")
    return pd.to_datetime(df.iloc[:, 1], errors="coerce", dayfirst=True)


def participant_dates_from_folder(folder: Path, kind: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    - participant (filename stem)
    - date (datetime.date)
    - kind ('counter' or 'journal')
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    rows = []
    for csv_path in sorted(folder.glob("*.csv")):
        participant = csv_path.stem  # one file = one participant
        df = read_csv_flexible(csv_path)

        if kind == "counter":
            dt = extract_dates_counter(df)
        elif kind == "journal":
            dt = extract_dates_journal(df)
        else:
            raise ValueError("kind must be 'counter' or 'journal'")

        # keep valid dates only
        dt = dt.dropna()
        if dt.empty:
            continue

        # unique dates for this participant (multiple entries on same date count once)
        for d in pd.Series(dt.dt.date).dropna().unique():
            rows.append((participant, d, kind))

    return pd.DataFrame(rows, columns=["participant", "date", "kind"])


def main() -> None:
    counter_pd = participant_dates_from_folder(COUNTER_DIR, "counter")
    journal_pd = participant_dates_from_folder(JOURNAL_DIR, "journal")

    # counts per day per dataset
    counter_daily = (
        counter_pd.groupby("date")["participant"].nunique().rename("counter_participants")
        if not counter_pd.empty
        else pd.Series(dtype="int64", name="counter_participants")
    )
    journal_daily = (
        journal_pd.groupby("date")["participant"].nunique().rename("journal_participants")
        if not journal_pd.empty
        else pd.Series(dtype="int64", name="journal_participants")
    )

    # total unique participants across both datasets per day
    # (treat counter and journal participants as distinct if same filename stem exists in both folders)
    both = pd.concat([counter_pd, journal_pd], ignore_index=True)
    if not both.empty:
        both["participant_global"] = both["kind"] + "::" + both["participant"]
        total_daily = both.groupby("date")["participant_global"].nunique().rename("total_participants")
    else:
        total_daily = pd.Series(dtype="int64", name="total_participants")

    # optional: overlap of same filename stems appearing in BOTH datasets on the same day
    # (helps if the same participant exists in output of both conditions)
    if not counter_pd.empty and not journal_pd.empty:
        overlap = (
            counter_pd.merge(journal_pd, on=["date", "participant"], how="inner")
            .groupby("date")["participant"].nunique()
            .rename("overlap_same_filename")
        )
    else:
        overlap = pd.Series(dtype="int64", name="overlap_same_filename")

    summary = pd.concat([counter_daily, journal_daily, total_daily, overlap], axis=1).fillna(0).astype(int)
    summary = summary.sort_index()
    summary.index = summary.index.astype("datetime64[ns]")  # nicer printing/sorting

    print(summary)
    summary.to_csv(OUT_CSV, index=True)  # comma-separated by default
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
