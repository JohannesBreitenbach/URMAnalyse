# plot_daily_active_rate_by_day_combined.py
# One combined diagram with two lines:
#   - Counter daily active rate
#   - Journal daily active rate
#
# Definitions:
# - One CSV file = one participant
# - Multiple entries on the same date for the same participant count only once
# - Daily active rate = (active participants that day) / (valid participants in that group)
#   where "valid participants" = participants with at least one parsable date
#
# X-axis: Day 1, Day 2, ... (based on the global earliest date across BOTH groups)
#
# Outputs:
#   - daily_active_rate_by_day_combined.png
#   - daily_active_rate_by_day_combined.pdf

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
COUNTER_DIR = BASE_DIR / "data_counter"
JOURNAL_DIR = BASE_DIR / "data_journal"

OUT_PNG = BASE_DIR / "daily_active_rate_by_day_combined.png"
OUT_PDF = BASE_DIR / "daily_active_rate_by_day_combined.pdf"


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
    # date expected in 2nd column (id, date, ...)
    if df.shape[1] < 2:
        raise ValueError("Journal CSV has < 2 columns; expected date in 2nd column.")
    return pd.to_datetime(df.iloc[:, 1], errors="coerce", dayfirst=True)


def participant_dates(folder: Path, kind: str) -> pd.DataFrame:
    """
    Returns DataFrame with unique (participant, date) pairs.
    date is normalized to midnight Timestamp (datetime64[ns]) for indexing.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    rows: list[tuple[str, pd.Timestamp]] = []
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

        # unique calendar days for this participant
        unique_days = pd.Series(dt.dt.normalize()).dropna().unique()
        for d in unique_days:
            rows.append((participant, pd.Timestamp(d)))

    out = pd.DataFrame(rows, columns=["participant", "date"]).drop_duplicates()
    return out


def daily_active_rate(pdays: pd.DataFrame) -> pd.Series:
    """
    Input: unique (participant, date)
    Output: Series indexed by date (Timestamp), value = active_rate (0..1)
    """
    if pdays.empty:
        return pd.Series(dtype="float64")

    valid_participants = pdays["participant"].nunique()
    daily_active = pdays.groupby("date")["participant"].nunique().sort_index()
    rate = daily_active / float(valid_participants)
    return rate


def to_day_index(rate: pd.Series, global_start: pd.Timestamp) -> pd.Series:
    """
    Convert date index to Day 1..N (int), based on global_start.
    """
    if rate.empty:
        return rate
    day_numbers = (rate.index.normalize() - global_start.normalize()).days + 1
    out = rate.copy()
    out.index = day_numbers.astype(int)
    return out


def main() -> None:
    counter_pdays = participant_dates(COUNTER_DIR, "counter")
    journal_pdays = participant_dates(JOURNAL_DIR, "journal")

    counter_rate = daily_active_rate(counter_pdays)
    journal_rate = daily_active_rate(journal_pdays)

    if counter_rate.empty and journal_rate.empty:
        print("No usable data found in either folder.")
        return

    # Global date range start/end for consistent Day 1..N mapping
    all_dates = []
    if not counter_rate.empty:
        all_dates.append(counter_rate.index.min())
        all_dates.append(counter_rate.index.max())
    if not journal_rate.empty:
        all_dates.append(journal_rate.index.min())
        all_dates.append(journal_rate.index.max())

    global_start = min(all_dates)
    global_end = max(all_dates)

    # Reindex to full date range so missing days show as 0
    full_range = pd.date_range(global_start.normalize(), global_end.normalize(), freq="D")

    counter_rate_full = counter_rate.reindex(full_range, fill_value=0.0)
    journal_rate_full = journal_rate.reindex(full_range, fill_value=0.0)

    counter_day = to_day_index(counter_rate_full, global_start)
    journal_day = to_day_index(journal_rate_full, global_start)

    # Plot (one diagram, two lines)
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(counter_day.index, counter_day.values, label="Counter (daily active rate)")
    ax.plot(journal_day.index, journal_day.values, label="Journal (daily active rate)")

    ax.set_xlabel("Study day")
    ax.set_ylabel("Daily active rate (active ÷ group size)")
    ax.set_ylim(0, 1)

    # Ticks: show "Day 1, Day 2, ..." but not for every single day if long
    max_day = int(max(counter_day.index.max(), journal_day.index.max()))
    if max_day <= 14:
        ticks = list(range(1, max_day + 1))
    else:
        step = 2 if max_day <= 30 else 5
        ticks = list(range(1, max_day + 1, step))
        if ticks[-1] != max_day:
            ticks.append(max_day)

    ax.set_xticks(ticks)
    ax.set_xticklabels([f"Day {t}" for t in ticks], rotation=30, ha="right")

    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
