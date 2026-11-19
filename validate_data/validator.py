"""Validate each streamed batch of detail/trx against the pandera schema.

Usage:
    python producer.py --seed 42 | python validator.py

Reads length-prefixed Arrow IPC batches from stdin until the stream ends.
Exits with code 1 if any batch failed.
"""
import argparse
import os
import struct
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import pandera.pandas as pa
import pyarrow as arrow

HERE = os.path.dirname(os.path.abspath(__file__))

# Keep integers/floats as pandas nullable types, so a batch with a missing
# value stays Int32 instead of silently turning into float64.
NULLABLE = {
    arrow.int32(): pd.Int32Dtype(),
    arrow.int64(): pd.Int64Dtype(),
    arrow.float32(): pd.Float32Dtype(),
    arrow.float64(): pd.Float64Dtype(),
}


def read_batches(stream):
    while True:
        header = stream.read(8)
        if not header:
            return
        (size,) = struct.unpack("<Q", header)
        yield arrow.ipc.open_stream(stream.read(size)).read_all()


def describe(err):
    """One short line per failed column/check."""
    fc = err.failure_cases
    lines = []
    for (col, check), grp in fc.groupby(["column", "check"], sort=False):
        sample = grp["failure_case"].iloc[0]
        lines.append(f"{col}: {check} (e.g. {sample!r}, {len(grp)} case(s))")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--schema", default=os.path.join(HERE, "trx_stream_schema.yml"))
    args = p.parse_args()

    schema = pa.DataFrameSchema.from_yaml(args.schema)
    passed = failed = rows = 0

    for i, table in enumerate(read_batches(sys.stdin.buffer)):
        df = table.to_pandas(types_mapper=NULLABLE.get)
        rows += len(df)
        try:
            schema.validate(df, lazy=True)
            passed += 1
            print(f"batch {i}  PASS")
        except pa.errors.SchemaErrors as err:
            failed += 1
            print(f"batch {i}  FAIL")
            for line in describe(err):
                print(f"    {line}")

    print(f"\n{passed + failed} batches, {rows:,} rows: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
