from pathlib import Path
import polars as pl

CUTOFF_MS = 1698796799000

# Use absolute paths based on script location
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent  # gaal -> students -> 494-user-trajectories

LOCAL_DATA = REPO_ROOT / "local-data"
OUT_ROOT = REPO_ROOT / "data" / "filtered" / "2023-10"

TIMESTAMP_COL = {
    "notes": "createdAtMillis",
    "notes-status-history": "createdAtMillis",
    "ratings": "createdAtMillis",
    # user-enrollment should NOT be filtered by timestamp - we need all enrolled users
    "notes-request-data": "noteRequestFeedEligibleAtMillis",
}

# Directories to copy without filtering
COPY_WITHOUT_FILTER = {"user-enrollment"}

for subdir in sorted(LOCAL_DATA.iterdir()):
    if not subdir.is_dir():
        continue

    out_dir = OUT_ROOT / subdir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy without filtering for certain directories
    if subdir.name in COPY_WITHOUT_FILTER:
        print(f"\nCopying {subdir.name}/ without filtering...")
        for tsv in sorted(subdir.glob("*.tsv")):
            out_path = out_dir / tsv.name
            lf = pl.scan_csv(tsv, separator="\t", infer_schema_length=0)
            lf.sink_csv(out_path, separator="\t")
            rows = lf.select(pl.len()).collect()[0, 0]
            print(f"  Copied {out_path} ({rows} rows)")
        continue

    ts_col = TIMESTAMP_COL.get(subdir.name)
    if ts_col is None:
        print(f"Skipping {subdir.name} (no timestamp column configured)")
        continue

    print(f"\nFiltering {subdir.name}/ on {ts_col} ...")

    for tsv in sorted(subdir.glob("*.tsv")):
        out_path = out_dir / tsv.name

        lf = pl.scan_csv(tsv, separator="\t", infer_schema_length=0)
        schema = lf.collect_schema().names()

        if ts_col not in schema:
            print(f"  Skipping {tsv.name} (no {ts_col} column)")
            continue

        filtered = (
            lf
            .with_columns(
                pl.col(ts_col)
                .cast(pl.Int64, strict=False)
                .alias(ts_col)
            )
            .filter(pl.col(ts_col).is_not_null())
            .filter(pl.col(ts_col) <= CUTOFF_MS)
        )

        filtered.sink_csv(out_path, separator="\t")

        rows = filtered.select(pl.len()).collect()[0, 0]
        print(f"  Wrote {out_path} ({rows} rows)")

print("\nDone.")
