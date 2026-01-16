# clean_all_counter_csvs.py
# Turns all CSVs in ./data_counter into "clean" CSVs in ./data_counter_clean
# Cleaning inferred from your example:
# - keep only: type, date
# - build date from day/month/year -> "DD.MM.YYYY"
# - sort by date descending (stable)

from __future__ import annotations

from pathlib import Path
import pandas as pd


IN_DIR = Path(__file__).resolve().parent / "data_counter"
OUT_DIR = Path(__file__).resolve().parent / "data_counter_clean"


def _clean_one_file(in_path: Path, out_dir: Path) -> Path:
    df = pd.read_csv(in_path)

    # Normalize column names (just in case)
    df.columns = [c.strip() for c in df.columns]

    if {"year", "month", "day"}.issubset(df.columns):
        dt = pd.to_datetime(
            dict(
                year=pd.to_numeric(df["year"], errors="coerce"),
                month=pd.to_numeric(df["month"], errors="coerce"),
                day=pd.to_numeric(df["day"], errors="coerce"),
            ),
            errors="coerce",
        )
        date_str = dt.dt.strftime("%d.%m.%Y")
    elif "date" in df.columns:
        # If already present, normalize to DD.MM.YYYY
        dt = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        date_str = dt.dt.strftime("%d.%m.%Y")
    else:
        raise ValueError(
            f"{in_path.name}: expected columns (year, month, day) or (date). Found: {list(df.columns)}"
        )

    if "type" not in df.columns:
        raise ValueError(f"{in_path.name}: missing required column 'type'.")

    out = pd.DataFrame({"type": df["type"], "date": date_str})
    # Drop invalid rows
    out = out.dropna(subset=["type", "date"])

    # Sort by date descending (stable)
    sort_dt = pd.to_datetime(out["date"], format="%d.%m.%Y", errors="coerce")
    out = out.assign(_sort_dt=sort_dt).sort_values("_sort_dt", ascending=False, kind="mergesort")
    out = out.drop(columns=["_sort_dt"])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}_clean.csv"
    out.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    return out_path


def main() -> None:
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {IN_DIR}")

    csv_files = sorted(IN_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in: {IN_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, 0
    for p in csv_files:
        try:
            out_path = _clean_one_file(p, OUT_DIR)
            print(f"OK   {p.name} -> {out_path.name}")
            ok += 1
        except Exception as e:
            print(f"FAIL {p.name}: {e}")
            failed += 1

    print(f"\nDone. Success: {ok}, Failed: {failed}")


if __name__ == "__main__":
    main()
