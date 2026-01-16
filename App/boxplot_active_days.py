# make_boxplot_active_days.py
# Creates a boxplot (and individual points) comparing "active day count" per participant
# for ./data_counter vs ./data_journal and saves it as an image file.
#
# Output file:
#   - active_days_boxplot.png
#
# Additionally prints the descriptive statistics that the plot reflects.

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
COUNTER_DIR = BASE_DIR / "data_counter"
JOURNAL_DIR = BASE_DIR / "data_journal"
OUT_PNG = BASE_DIR / "active_days_boxplot.png"


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


def participant_day_counts(folder: Path, kind: str) -> pd.DataFrame:
    """Return: participant | active_days"""
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    rows: list[tuple[str, int]] = []
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
        active_days = pd.Series(dt.dt.date).dropna().nunique()
        rows.append((participant, int(active_days)))

    return pd.DataFrame(rows, columns=["participant", "active_days"])


def print_stats(label: str, vals: pd.Series) -> None:
    vals = vals.dropna().astype(float)
    n = int(vals.shape[0])

    if n == 0:
        print(f"\n{label}: no valid values.")
        return

    mean = vals.mean()
    median = vals.median()
    sd = vals.std(ddof=1) if n > 1 else 0.0
    q1 = vals.quantile(0.25)
    q3 = vals.quantile(0.75)
    iqr = q3 - q1
    vmin = vals.min()
    vmax = vals.max()

    print(f"\n{label}")
    print(f"n = {n}")
    print(f"values = {', '.join(str(int(v)) if float(v).is_integer() else str(v) for v in vals.tolist())}")
    print(f"mean = {mean:.2f}")
    print(f"median = {median:.2f}")
    print(f"sd = {sd:.2f}")
    print(f"min = {vmin:.2f}")
    print(f"q1 (25%) = {q1:.2f}")
    print(f"q3 (75%) = {q3:.2f}")
    print(f"iqr = {iqr:.2f}")
    print(f"max = {vmax:.2f}")


def main() -> None:
    counter = participant_day_counts(COUNTER_DIR, "counter").assign(group="Counter")
    journal = participant_day_counts(JOURNAL_DIR, "journal").assign(group="Journal")
    data = pd.concat([counter, journal], ignore_index=True)

    if data.empty:
        print("No data found. Check that data_counter/ and data_journal/ contain CSV files.")
        return

    # Print descriptive stats (what the boxplot summarizes)
    print_stats("Counter group (active days per participant)", data.loc[data["group"] == "Counter", "active_days"])
    print_stats("Journal group (active days per participant)", data.loc[data["group"] == "Journal", "active_days"])

    # Prepare data for matplotlib
    groups = ["Counter", "Journal"]
    values = [
        data.loc[data["group"] == g, "active_days"].astype(float).tolist()
        for g in groups
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(values, labels=groups, showfliers=True)

    # Show individual participant points (helpful with small n)
    for i, g in enumerate(groups, start=1):
        ys = data.loc[data["group"] == g, "active_days"].astype(float).values
        xs = [i] * len(ys)
        ax.scatter(xs, ys, alpha=0.7)

    ax.set_ylabel("Active days per participant")
    ax.set_title("Engagement by app type (unique active days)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
