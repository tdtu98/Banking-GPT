**Banking GPT**

This repo aims to build a complete flow for pre-training, fine-tuning and serving a foundation model on banking domain.

**Plan**
- https://docs.google.com/spreadsheets/d/1p9K_GhfJmuSfJgssHxk2U8KZT0OCgGcqunqfN5ASPF8/edit?gid=1006#gid=1006

**Data**
We get data from these sources:

- https://huggingface.co/datasets/ai-lab/MBD
- https://huggingface.co/datasets/ai-lab/MBD-mini

To download data:

```bash
cd data
bash script.sh
```

**Data checks**

Every MBD table has a Pandera schema in `data/schemas.py` (columns, types and value rules, derived from profiling the real MBD-mini data). All 3,820 real files pass. `validate_batch()` is the arrival gate: it checks one incoming batch and returns `None` if it passes, or the failing rows if not.

```python
import polars as pl
from schemas import validate_batch   # run from data/, or add data/ to sys.path

failures = validate_batch(pl.read_parquet("mbd_mini/detail/trx/fold=0/part-....parquet"), "detail/trx")
print("PASS" if failures is None else failures)
```

Tables: `detail/trx`, `detail/geo`, `detail/dialog`, `targets`, `client_split`, `ptls/trx`, `ptls/geo`, `ptls/dialog`.

`sim/simulate_arrival.py` simulates data arriving from an upstream system. It takes real MBD files as incoming batches, breaks some of them on purpose in memory (wrong type, null ids, unknown codes, extra or missing columns, timezone changes, empty batches, unsorted events and more), and sends each one through the gate. The files on disk are never changed.

```bash
pip install -r requirements.txt
python sim/simulate_arrival.py                                   # all 8 tables
python sim/simulate_arrival.py --table detail/trx --delay 0.3    # one table, slowly
```

Sample output (real run, seed 42):

```
=== detail/trx  (Trx schema v1.1.0)
[batch 01]  10000 rows  ACCEPT      sent: clean
[batch 02]  10000 rows  QUARANTINE  sent: timezone_change
             ↳ event_time: expected Datetime(time_unit='ns', time_zone=None), got Datetime(time_unit='ns', time_zone='UTC')
[batch 04]  10000 rows  QUARANTINE  sent: unknown_currency
             ↳ currency: isin([1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 15, 16, 17]) failed on 10 rows
[batch 06]  10000 rows  QUARANTINE  sent: wrong_type
             ↳ amount: expected Float32, got String
[batch 09]  10000 rows  QUARANTINE  sent: extra_column
             ↳ unexpected column 'channel'
[batch 11]      0 rows  QUARANTINE  sent: empty_batch
             ↳ batch is empty
[batch 12]  10000 rows  ACCEPT      sent: clean
...

table           accepted quarantined missed false alarm
detail/trx             2          10      0           0
detail/geo             2           9      0           0
detail/dialog          2           9      0           0
targets                2           8      0           0
client_split           2           7      0           0
ptls/trx               2          10      0           0
ptls/geo               2           9      0           0
ptls/dialog            2           9      0           0
```

- **ACCEPT**: the batch passed every rule and can be used.
- **QUARANTINE**: the batch was rejected; the lines under it give the column, the rule and how many rows broke it.
- **missed**: a broken batch was accepted, so a rule is missing.
- **false alarm**: a clean batch was rejected, so a rule is too strict.

The script exits with an error if anything is missed or falsely rejected. Each batch is logged to `sim/gate_log.jsonl`.

What the real data showed while writing the rules:

- `fold` is not stored inside the parquet files, only in the folder names (`fold=0`…). Read with `hive_partitioning=True` to get it.
- 118 of 38.7M transactions have a null `currency`. The schema allows it for now (open decision).
- In `targets`, `diff_trans_date` is null exactly when `trans_count == 0` (10% of rows). This is a rule in the schema.
- `ptls` is a different encoding from `detail`: no nulls inside lists, `event_time` in unix seconds, at most 10,000 transactions per client.
- `client_split` has 100,224 clients, but only 98,721 appear in `ptls/trx`.
- Geo data starts on 2021-12-11 (about one year); transactions cover two years. Timestamps look like UTC+3 midnight stored without a timezone.
- Code ranges such as `event_type` 1–56 come from MBD-mini. The full MBD may contain codes the gate would reject (open decision).

These checks cover the format and values of the data. They don't check that no file is missing or corrupted; that needs a checksum manifest.

**Reference**
- https://arxiv.org/html/2611.03302v1
- https://github.com/deburky/EventGPT
- https://arxiv.org/abs/2409.17587