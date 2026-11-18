"""Pandera data contracts for all MBD tables.

Rules are derived from profiling the real MBD-mini data (2026-11-18):

  table          files       rows   notes
  detail/trx       400  38.7M       src/dst features 0.1-0.6% null, 118 null currency
  detail/geo       460  66.3M       starts 2021-12-11 (only ~1 year of geo)
  detail/dialog    415   217k       embedding = 768 floats
  targets          435   1.2M       diff_trans_date null <=> trans_count == 0
  client_split    1000   100k       one row per client
  ptls/trx         335  98.7k       one row per client, lists of events (max 10,000)
  ptls/geo         365  72.6k       one row per client
  ptls/dialog      410  46.3k       one row per client, list of 768-d embeddings

`fold` is NOT stored inside the parquet files: it only exists in the folder
name (fold=N), so these schemas validate the file contents without it.

Bump SCHEMA_VERSION whenever any rule changes.
"""
from datetime import datetime

import pandera.polars as pa
import polars as pl
from pandera.engines.polars_engine import DateTime, List

SCHEMA_VERSION = "1.1.0"

CLIENT_ID = r"^[0-9a-f]{64}$"  # all ids are 64-char hex hashes

# Currency codes seen in MBD-mini. A code outside this list = unknown upstream change.
KNOWN_CURRENCIES = [1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 15, 16, 17]

# All events fall in this window (stored as naive timestamps, ~UTC+3 midnight).
T_MIN, T_MAX = datetime(2020, 12, 31), datetime(2023, 1, 1)
# Same window as unix seconds, used by the ptls tables.
EPOCH_MIN, EPOCH_MAX = 1609372800, 1672531200

EMBEDDING_DIM = 768
NAIVE_NS = {"time_unit": "ns", "time_zone": None}


def _event_time():
    return pa.Field(nullable=False, dtype_kwargs=NAIVE_NS, ge=T_MIN, lt=T_MAX)


class _Strict(pa.DataFrameModel):
    """Shared config: every table must match its columns and types exactly."""

    class Config:
        strict = True   # unexpected / missing columns fail
        coerce = False  # a type change upstream must fail, not be silently converted

    @pa.dataframe_check(error="batch is empty")
    def not_empty(cls, data) -> pl.LazyFrame:
        return data.lazyframe.select(pl.len() > 0)


# ----------------------------------------------------------------- detail (one row per event)
class Trx(_Strict):
    client_id: pl.String = pa.Field(nullable=False, str_matches=CLIENT_ID)
    event_time: DateTime = _event_time()
    amount: pl.Float32 = pa.Field(nullable=False, gt=0)  # min in data is > 0
    event_type: pl.Int32 = pa.Field(nullable=False, in_range={"min_value": 1, "max_value": 56})
    event_subtype: pl.Int32 = pa.Field(nullable=False, in_range={"min_value": 1, "max_value": 62})
    # 118 of 38.7M real rows have a null currency, so nulls are allowed (policy decision).
    currency: pl.Int32 = pa.Field(nullable=True, isin=KNOWN_CURRENCIES)
    # src/dst features are optional in the real data (0.1%-0.6% nulls).
    src_type11: pl.Int32 = pa.Field(nullable=True, ge=1)
    src_type12: pl.Int32 = pa.Field(nullable=True, ge=1)
    dst_type11: pl.Int32 = pa.Field(nullable=True, ge=1)
    dst_type12: pl.Int32 = pa.Field(nullable=True, ge=1)
    src_type21: pl.Int32 = pa.Field(nullable=True, ge=1)
    src_type22: pl.Int32 = pa.Field(nullable=True, ge=1)
    src_type31: pl.Int32 = pa.Field(nullable=True, ge=1)
    src_type32: pl.Int32 = pa.Field(nullable=True, ge=1)


class Geo(_Strict):
    client_id: pl.String = pa.Field(nullable=False, str_matches=CLIENT_ID)
    event_time: DateTime = _event_time()
    geohash_4: pl.Int32 = pa.Field(nullable=False, ge=1)
    geohash_5: pl.Int32 = pa.Field(nullable=False, ge=1)
    geohash_6: pl.Int32 = pa.Field(nullable=False, ge=1)


class Dialog(_Strict):
    client_id: pl.String = pa.Field(nullable=False, str_matches=CLIENT_ID)
    event_time: DateTime = _event_time()
    embedding: List = pa.Field(nullable=False, dtype_kwargs={"inner": pl.Float32})

    @pa.check("embedding", error=f"embedding must have exactly {EMBEDDING_DIM} values")
    def embedding_dim(cls, data) -> pl.LazyFrame:
        return data.lazyframe.select(pl.col(data.key).list.len() == EMBEDDING_DIM)

    @pa.check("embedding", error="embedding contains null or NaN values")
    def embedding_finite(cls, data) -> pl.LazyFrame:
        return data.lazyframe.select(
            pl.col(data.key).list.eval(pl.element().is_null() | pl.element().is_nan())
            .list.any().not_())


# ----------------------------------------------------------------- targets / folds
class Targets(_Strict):
    client_id: pl.String = pa.Field(nullable=False, str_matches=CLIENT_ID)
    mon: pl.String = pa.Field(nullable=False, str_matches=r"^\d{4}-\d{2}-\d{2}$")
    target_1: pl.Int32 = pa.Field(isin=[0, 1])
    target_2: pl.Int32 = pa.Field(isin=[0, 1])
    target_3: pl.Int32 = pa.Field(isin=[0, 1])
    target_4: pl.Int32 = pa.Field(isin=[0, 1])
    trans_count: pl.Int32 = pa.Field(ge=0)
    # Null in ~10% of real rows: exactly the rows with trans_count == 0.
    diff_trans_date: pl.Int32 = pa.Field(nullable=True, in_range={"min_value": 0, "max_value": 366})

    @pa.dataframe_check(error="diff_trans_date must be null if and only if trans_count == 0")
    def gap_null_iff_no_trx(cls, data) -> pl.LazyFrame:
        return data.lazyframe.select(
            pl.col("diff_trans_date").is_null() == (pl.col("trans_count") == 0))


class ClientSplit(_Strict):
    client_id: pl.String = pa.Field(nullable=False, unique=True, str_matches=CLIENT_ID)


# ----------------------------------------------------------------- ptls (one row per client)
class _Ptls(_Strict):
    """pytorch-lifestream format: one row per client, each feature is a list of
    events in time order. All lists in a row must have the same length."""

    client_id: pl.String = pa.Field(nullable=False, unique=True, str_matches=CLIENT_ID)
    event_time: List = pa.Field(nullable=False, dtype_kwargs={"inner": pl.Int64})  # unix seconds

    @pa.check("event_time", error="event_time outside the expected time window")
    def event_time_window(cls, data) -> pl.LazyFrame:
        c = pl.col(data.key)
        return data.lazyframe.select((c.list.min() >= EPOCH_MIN) & (c.list.max() < EPOCH_MAX))

    @pa.check("event_time", error="events are not sorted by time")
    def event_time_sorted(cls, data) -> pl.LazyFrame:
        return data.lazyframe.select(
            pl.col(data.key).list.eval(pl.element().diff().drop_nulls().ge(0).all()).list.first())

    @pa.dataframe_check(error="all event lists in a row must have the same length")
    def same_length(cls, data) -> pl.LazyFrame:
        schema = data.lazyframe.collect_schema()
        lists = [c for c, t in schema.items() if isinstance(t, pl.List) and c != "event_time"]
        n = pl.col("event_time").list.len()
        return data.lazyframe.select(pl.all_horizontal([pl.col(c).list.len() == n for c in lists]))

    @pa.dataframe_check(error="event lists contain nulls or codes < 1")
    def codes_present(cls, data) -> pl.LazyFrame:
        schema = data.lazyframe.collect_schema()
        codes = [c for c, t in schema.items()
                 if isinstance(t, pl.List) and t.inner == pl.Int32]
        return data.lazyframe.select(pl.all_horizontal(
            [(~pl.col(c).list.eval(pl.element().is_null()).list.any())
             & (pl.col(c).list.min() >= 1) for c in codes] or [pl.lit(True)]))


def _int_list():
    return pa.Field(nullable=False, dtype_kwargs={"inner": pl.Int32})


class PtlsTrx(_Ptls):
    # In ptls, detail's nulls are re-encoded as codes, so inner values are never null.
    amount: List = pa.Field(nullable=False, dtype_kwargs={"inner": pl.Float32})
    event_type: List = _int_list()
    event_subtype: List = _int_list()
    currency: List = _int_list()
    src_type11: List = _int_list()
    src_type12: List = _int_list()
    dst_type11: List = _int_list()
    dst_type12: List = _int_list()
    src_type21: List = _int_list()
    src_type22: List = _int_list()
    src_type31: List = _int_list()
    src_type32: List = _int_list()

    @pa.check("event_time", error="more than 10,000 events per client")
    def max_events(cls, data) -> pl.LazyFrame:
        return data.lazyframe.select(pl.col(data.key).list.len() <= 10_000)

    @pa.check("amount", error="amount must be > 0")
    def amount_positive(cls, data) -> pl.LazyFrame:
        return data.lazyframe.select(pl.col(data.key).list.min() > 0)


class PtlsGeo(_Ptls):
    geohash_4: List = _int_list()
    geohash_5: List = _int_list()
    geohash_6: List = _int_list()


class PtlsDialog(_Ptls):
    embedding: List = pa.Field(nullable=False, dtype_kwargs={"inner": pl.List(pl.Float32)})

    @pa.check("embedding", error=f"every embedding must have exactly {EMBEDDING_DIM} values")
    def embedding_dim(cls, data) -> pl.LazyFrame:
        return data.lazyframe.select(
            pl.col(data.key).list.eval(pl.element().list.len() == EMBEDDING_DIM).list.all())


TABLES: dict[str, type[pa.DataFrameModel]] = {
    "detail/trx": Trx,
    "detail/geo": Geo,
    "detail/dialog": Dialog,
    "targets": Targets,
    "client_split": ClientSplit,
    "ptls/trx": PtlsTrx,
    "ptls/geo": PtlsGeo,
    "ptls/dialog": PtlsDialog,
}


def validate_batch(df: pl.DataFrame, table: str) -> pl.DataFrame | None:
    """Validate one arriving batch. Returns None if it passes, else the failure cases."""
    try:
        TABLES[table].validate(df, lazy=True)
        return None
    except pa.errors.SchemaErrors as e:
        return e.failure_cases
