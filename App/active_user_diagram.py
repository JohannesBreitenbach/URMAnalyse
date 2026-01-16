# plot_daily_active_counts.py
# Generates a line chart of DAILY ACTIVE PARTICIPANTS (unique per file/participant per day)
# for ./data_counter and ./data_journal.
#
# Output files:
#   - daily_active_counts_counter.png / .pdf
#   - daily_active_counts_journal.png / .pdf
#
# Notes:
# - One file = one participant
# - Multiple entries on same date for same participant count once
# - Y-axis: user count (active participants)
# - X-axis: dates

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
COUNTER_DIR = BASE_DIR / "data_counter"
JOURNAL_DIR = BASE_DIR / "data_journal"


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Try common separators, then fall back to sniffing."""
    for sep in [",", ";", "\t", "|"]:
        try:
            return pd.read_csv(path, sep=sep)
        except Exception:
            pass
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

    if "date" in cols:
        return pd.to_datetime(df[cols["date"]], errors="coerce", dayfirst=True)

    # fallback: try first column as date
    return pd.to_datetime(df.iloc[:, 0], errors="coerce", dayfirst=True)


def extract_dates_journal(df: pd.DataFrame) -> pd.Series:
    # date is expected in the 2nd column (id, date, ...)
    if df.shape[1] < 2:
        raise ValueError("Journal CSV has < 2 columns; expected date in 2nd column.")
    return pd.to_datetime(df.iloc[:, 1], errors="coerce", dayfirst=True)


def participant_dates_from_folder(folder: Path, kind: str) -> pd.DataFrame:
    """
    Returns DataFrame:
      participant | date
    where date is a Python date (no time), and only unique (participant, date) pairs exist.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    rows = []
    for csv_path in sorted(folder.glob("*.csv")):
        participant = csv_path.stem
        df = read_csv_flexible(csv_path)

        if kind == "counter":
            dt = extract_dates_counter(df)
        elif kind == "journal":
            dt = extract_dates_journal(df)
        else:
            raise ValueError("kind must be 'counter' or 'journal'")

        dt = dt.dropna()
        if dt.empty:
            continue

        unique_days = pd.Series(dt.dt.date).dropna().unique()
        for d in unique_days:
            rows.append((participant, d))

    out = pd.DataFrame(rows, columns=["participant", "date"])
    out = out.drop_duplicates()
    return out


def daily_active_counts(folder: Path, kind: str) -> pd.Series:
    """
    Returns a Series indexed by datetime (daily) with values = number of active participants that day.
    """
    pdays = participant_dates_from_folder(folder, kind)
    if pdays.empty:
        return pd.Series(dtype="int64")

    s = pdays.groupby("date")["participant"].nunique().sort_index()
    # convert index to datetime for plotting
    s.index = pd.to_datetime(s.index)
    return s


def plot_series(s: pd.Series, title: str, out_base: Path) -> None:
    if s.empty:
        print(f"{title}: no data found.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s.index, s.values)  # no custom colors specified

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Active participants (count)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()

    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)

    print(f"Saved: {out_base.with_suffix('.png')}")
    print(f"Saved: {out_base.with_suffix('.pdf')}")


def main() -> None:
    counter = daily_active_counts(COUNTER_DIR, "counter")
    journal = daily_active_counts(JOURNAL_DIR, "journal")

    plot_series(counter, "Daily active participants — Counter", BASE_DIR / "daily_active_counts_counter")
    plot_series(journal, "Daily active participants — Journal", BASE_DIR / "daily_active_counts_journal")


if __name__ == "__main__":
    main()
