"""Simulate data arriving from an upstream bank system and being checked at arrival.

Each MBD parquet file plays the role of one incoming batch. Some batches are
deliberately broken to mimic real upstream problems. Every batch goes through
the Pandera arrival gate (data/schemas.py) and gets ACCEPTED or QUARANTINED.

Nothing is copied: batches are read from data/mbd_mini, changed in memory,
and only a small log is written (sim/gate_log.jsonl).

    python sim/simulate_arrival.py                       # all 8 tables
    python sim/simulate_arrival.py --table detail/trx    # one table
    python sim/simulate_arrival.py --delay 0.3           # watch batches "arrive"
"""
import argparse
import glob
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "data"))
from schemas import SCHEMA_VERSION, TABLES, validate_batch  # noqa: E402

DATA_ROOT = REPO / "data" / "mbd_mini"
LOG_PATH = REPO / "sim" / "gate_log.jsonl"
MODEL_NAMES = {m.__name__ for m in TABLES.values()}


def every(n):
    """Mask selecting every n-th row (always includes row 0)."""
    return pl.int_range(pl.len()) % n == 0


def value_col(df):
    """First feature column (not the key or timestamp)."""
    rest = [c for c in df.columns if c not in ("client_id", "event_time")]
    return rest[0] if rest else df.columns[0]


# --- problems any table can receive ---------------------------------------------
def clean(df):
    return df

def wrong_type(df):           # upstream changes a column's type
    c = value_col(df)
    t = df.schema[c]
    if isinstance(t, pl.List) and isinstance(t.inner, pl.List):
        new = pl.List(pl.List(pl.Float64))   # nested: list of embeddings
    elif isinstance(t, pl.List):
        new = pl.List(pl.Float64) if t.inner == pl.Float32 else pl.List(pl.Int64)
    elif t == pl.String:
        new = pl.Categorical
    else:
        new = pl.String
    return df.with_columns(pl.col(c).cast(new))

def null_client_id(df):       # some rows lose their client id
    return df.with_columns(pl.when(every(50)).then(None).otherwise(pl.col("client_id")).alias("client_id"))

def bad_client_id(df):        # ids arrive in a different format (not 64-char hex)
    return df.with_columns(pl.when(every(50)).then(pl.lit("CLIENT-0001"))
                           .otherwise(pl.col("client_id")).alias("client_id"))

def extra_column(df):         # upstream adds a column
    return df.with_columns(pl.lit("web").alias("channel"))

def missing_column(df):       # upstream drops a column
    return df.drop(df.columns[-1])

def empty_batch(df):          # export job ran but produced nothing
    return df.head(0)

COMMON = [wrong_type, null_client_id, bad_client_id, extra_column, missing_column, empty_batch]


# --- detail tables (one row per event) ------------------------------------------
def timezone_change(df):      # timestamps start arriving as UTC-aware
    return df.with_columns(pl.col("event_time").dt.replace_time_zone("UTC"))

def future_dates(df):         # clock / parsing bug upstream
    return df.with_columns(pl.col("event_time").dt.offset_by("3y"))

def unknown_currency(df):     # a new currency code nobody told us about
    return df.with_columns(pl.when(every(1000)).then(pl.lit(99, pl.Int32))
                           .otherwise(pl.col("currency")).alias("currency"))

def negative_amount(df):      # sign convention changes upstream
    return df.with_columns(-pl.col("amount"))

def zero_geohash(df):         # missing location sent as 0 instead of a real cell
    return df.with_columns(pl.when(every(100)).then(pl.lit(0, pl.Int32))
                           .otherwise(pl.col("geohash_6")).alias("geohash_6"))

def short_embedding(df):      # upstream switches to a smaller embedding model
    return df.with_columns(pl.col("embedding").list.head(512))

def nan_embedding(df):        # embedding service fails for some dialogs
    return df.with_columns(pl.when(every(10)).then(pl.col("embedding").list.eval(pl.element() * float("nan")))
                           .otherwise(pl.col("embedding")).alias("embedding"))


# --- targets / client_split -----------------------------------------------------
def invalid_target(df):       # label encoding changes (e.g. 2 = "unknown")
    return df.with_columns(pl.when(every(100)).then(pl.lit(2, pl.Int32))
                           .otherwise(pl.col("target_1")).alias("target_1"))

def gap_rule_broken(df):      # diff_trans_date missing although there were transactions
    return df.with_columns(pl.when(pl.col("trans_count") > 0).then(None)
                           .otherwise(pl.col("diff_trans_date")).alias("diff_trans_date"))

def duplicate_clients(df):    # same client exported twice
    return pl.concat([df, df.head(5)])


# --- ptls tables (one row per client, lists of events) --------------------------
def unsorted_events(df):      # events no longer in time order
    return df.with_columns(pl.col("event_time").list.reverse())

def length_mismatch(df):      # one feature list loses its last event
    c = value_col(df)
    return df.with_columns(pl.when(pl.col(c).list.len() > 1).then(pl.col(c).list.head(pl.col(c).list.len() - 1))
                           .otherwise(pl.col(c)).alias(c))

def epoch_millis(df):         # event_time switches from seconds to milliseconds
    return df.with_columns(pl.col("event_time").list.eval(pl.element() * 1000))

def null_code_in_list(df):    # a missing code arrives as null inside the list
    return df.with_columns(pl.col("event_type").list.eval(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.element())))

def short_embeddings_ptls(df):
    return df.with_columns(pl.col("embedding").list.eval(pl.element().list.head(512)))


PROBLEMS = {
    "detail/trx": [timezone_change, future_dates, unknown_currency, negative_amount],
    "detail/geo": [timezone_change, future_dates, zero_geohash],
    "detail/dialog": [timezone_change, short_embedding, nan_embedding],
    "targets": [invalid_target, gap_rule_broken],
    "client_split": [duplicate_clients],
    "ptls/trx": [unsorted_events, length_mismatch, epoch_millis, null_code_in_list],
    "ptls/geo": [unsorted_events, length_mismatch, epoch_millis],
    "ptls/dialog": [unsorted_events, length_mismatch, short_embeddings_ptls],
}


def schedule(table):
    """Clean batch first and last, every problem once in between."""
    return [clean] + PROBLEMS[table] + COMMON + [clean]


def summarize(failures: pl.DataFrame) -> list[str]:
    """Turn Pandera failure cases into short human-readable reasons."""
    # If a column has the wrong type, its value checks are just noise: drop them.
    bad_type = failures.filter(pl.col("check").str.starts_with("dtype("))["column"].to_list()
    failures = failures.filter(
        ~pl.col("column").is_in(bad_type) | pl.col("check").str.starts_with("dtype("))

    reasons = []
    for r in (failures.group_by("column", "check", maintain_order=True)
              .agg(pl.len().alias("n"), pl.col("failure_case").first())
              .iter_rows(named=True)):
        if r["check"] == "column_in_schema":
            reasons.append(f"unexpected column '{r['failure_case']}'")
        elif r["check"] == "column_in_dataframe":
            reasons.append(f"missing column '{r['failure_case']}'")
        elif r["check"].startswith("dtype("):
            reasons.append(f"{r['column']}: expected {r['check'][7:-2]}, got {r['failure_case']}")
        elif r["column"] in MODEL_NAMES:   # check on the whole batch
            reasons.append(r["check"] if r["n"] == 1 else f"{r['check']} ({r['n']} rows)")
        else:
            reasons.append(f"{r['column']}: {r['check']} failed on {r['n']} rows")
    return reasons


def run_table(table, args, log, rng):
    files = sorted(glob.glob(str(DATA_ROOT / table / "fold=*" / "*.parquet")))
    if not files:
        sys.exit(f"No files under {DATA_ROOT / table}. Run data/script.sh first.")
    plan = schedule(table)
    sources = rng.sample(files, min(len(plan), len(files)))

    print(f"\n=== {table}  ({TABLES[table].__name__} schema v{SCHEMA_VERSION})")
    stats = {"accepted": 0, "quarantined": 0, "missed": 0, "false_alarm": 0}
    for i, (problem, src) in enumerate(zip(plan, sources), 1):
        batch = problem(pl.read_parquet(src).head(args.rows))
        failures = validate_batch(batch, table)
        ok = failures is None
        reasons = [] if ok else summarize(failures)

        stats["accepted" if ok else "quarantined"] += 1
        if ok and problem is not clean:
            stats["missed"] += 1          # broken batch got through: gate has a gap
        if not ok and problem is clean:
            stats["false_alarm"] += 1     # good batch rejected: rule too strict

        status = "ACCEPT    " if ok else "QUARANTINE"
        flag = "  <-- MISSED" if ok and problem is not clean else ""
        print(f"[batch {i:02d}] {batch.height:>6} rows  {status}  sent: {problem.__name__}{flag}")
        for r in reasons:
            print(f"             ↳ {r}")

        log.write(json.dumps({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "table": table, "batch": i, "source_file": Path(src).name,
            "rows": batch.height, "schema_version": SCHEMA_VERSION,
            "simulated_problem": problem.__name__,
            "status": "accepted" if ok else "quarantined", "reasons": reasons,
        }) + "\n")
        time.sleep(args.delay)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="all", choices=["all", *TABLES])
    p.add_argument("--rows", type=int, default=10_000, help="max rows per batch")
    p.add_argument("--delay", type=float, default=0.0, help="seconds between arrivals")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tables = list(TABLES) if args.table == "all" else [args.table]
    rng = random.Random(args.seed)
    results = {}
    with open(LOG_PATH, "a") as log:
        for t in tables:
            results[t] = run_table(t, args, log, rng)

    print(f"\n{'table':15} {'accepted':>8} {'quarantined':>11} {'missed':>6} {'false alarm':>11}")
    for t, s in results.items():
        print(f"{t:15} {s['accepted']:>8} {s['quarantined']:>11} {s['missed']:>6} {s['false_alarm']:>11}")
    print(f"\nLog: {LOG_PATH.relative_to(REPO)}")
    if any(s["missed"] or s["false_alarm"] for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
