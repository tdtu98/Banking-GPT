"""Fake stream: read detail/trx with PySpark and send batches to stdout.

Usage:
    python producer.py --seed 42 | python validator.py

Each batch is written as its own Arrow IPC stream, prefixed with an 8-byte
length, so a corrupted batch may carry different column types than the others.
stdout carries only data; all messages go to stderr.
"""
import argparse
import os
import random
import struct
import sys

import pyarrow as pa
import pyarrow.compute as pc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "mbd_mini", "detail", "trx")
BREW_JAVA = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"


def log(msg):
    print(f"[producer] {msg}", file=sys.stderr, flush=True)


# Each corruption changes the data type of one column in the batch.
def amount_to_string(t):
    return "amount", pc.cast(t["amount"], pa.string())

def event_type_to_float(t):
    return "event_type", pc.cast(t["event_type"], pa.float64())

def event_time_to_string(t):
    return "event_time", pc.cast(t["event_time"], pa.string())

def client_id_to_int(t):
    # Take the first 8 hex chars of the id and turn them into a number.
    ids = [int(x[:8], 16) for x in t["client_id"].to_pylist()]
    return "client_id", pa.array(ids, pa.int64())

def event_subtype_to_int64(t):
    # Subtle: still an integer, but 64-bit instead of the contract's 32-bit.
    return "event_subtype", pc.cast(t["event_subtype"], pa.int64())

CORRUPTIONS = {
    "amount -> string": amount_to_string,
    "event_type -> float64": event_type_to_float,
    "event_time -> string": event_time_to_string,
    "client_id -> int64": client_id_to_int,
    "event_subtype -> int64": event_subtype_to_int64,
}


def corrupt(table, rng):
    name = rng.choice(list(CORRUPTIONS))
    col, new = CORRUPTIONS[name](table)
    return table.set_column(table.schema.get_field_index(col), col, new), name


def send(table, out):
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    buf = sink.getvalue()
    out.write(struct.pack("<Q", buf.size))
    out.write(buf)
    out.flush()


def read_with_spark(fold, n_rows):
    if not os.environ.get("JAVA_HOME") and os.path.isdir(BREW_JAVA):
        os.environ["JAVA_HOME"] = BREW_JAVA
    from pyspark.sql import SparkSession

    spark = (SparkSession.builder.master("local[*]").appName("trx-fake-stream")
             .config("spark.ui.enabled", "false")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    # fold is a partition column (fold=1/ folders), so Spark only reads that
    # folder: partition pruning. Sorting makes the batches reproducible.
    df = (spark.read.parquet(DATA)
          .filter(f"fold = {fold}")
          .orderBy("client_id", "event_time")
          .limit(n_rows))
    table = df.toArrow()
    spark.stop()
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--batches", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--error-rate", type=float, default=0.3,
                   help="probability that a batch gets a type error")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    rng = random.Random(args.seed)
    log(f"reading fold {args.fold} with Spark ...")
    table = read_with_spark(args.fold, args.batches * args.batch_size)
    log(f"got {table.num_rows:,} rows, streaming {args.batches} batches")

    out = sys.stdout.buffer
    for i in range(args.batches):
        batch = table.slice(i * args.batch_size, args.batch_size)
        if rng.random() < args.error_rate:
            batch, what = corrupt(batch, rng)
            log(f"batch {i}: CORRUPTED {what}")
        else:
            log(f"batch {i}: clean")
        send(batch, out)
    log("done")


if __name__ == "__main__":
    main()
