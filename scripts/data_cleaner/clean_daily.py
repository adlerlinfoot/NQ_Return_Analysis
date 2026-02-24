import pandas as pd
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

def clean_barchart_daily(path):
    # Read as standard CSV
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Rename to standard names
    df = df.rename(columns={
        "time": "date",
        "latest": "close"
    })

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Sort and dedupe
    df = df.sort_values("date").drop_duplicates("date")

    # Convert close to float (in case it's string)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Compute returns
    df["return"] = df["close"].pct_change().fillna(0)

    return df[["date", "close", "return"]]

# Clean both datasets
nq = clean_barchart_daily(RAW / "NQ.csv")
gc = clean_barchart_daily(RAW / "GC.csv")

# Align calendars
merged = pd.merge(nq, gc, on="date", how="inner", suffixes=("_nq", "_gc"))

# Save
merged[["date", "close_nq", "return_nq"]].to_csv(PROC / "nq_clean.csv", index=False)
merged[["date", "close_gc", "return_gc"]].to_csv(PROC / "gc_clean.csv", index=False)

print("Saved cleaned NQ and GC daily series.")