# validate_data

Data type validation for the MBD-mini transactions table (`data/mbd_mini/detail/trx`) on a simulated stream.

## Target

- Simulate how data arrives in a real pipeline: a **producer** reads the data with **PySpark** and sends it as a stream of small batches.
- Check **every batch** against a data contract (a **pandera** schema in YAML) before the data is used.
- Inject **random data type errors** on purpose, to prove the validator catches them and to show what each failure looks like.

The current setup streams **10 batches × 1,000 rows from fold 1**, with a 30% chance that a batch gets a type error.

```
producer.py (PySpark)  ──pipe──▶  validator.py (pandas + pandera)
reads fold 1                       validates each batch against trx_stream_schema.yml
10 batches × 1,000 rows            prints PASS / FAIL per batch + summary
random type errors
```

## File structure

```
project/
├── data/
│   ├── download_data.py        # downloads MBD-mini from Hugging Face
│   ├── script.sh               # download + extract all archives
│   └── mbd_mini/
│       └── detail/trx/         # input: transactions, partitioned as fold=0/ … fold=4/
└── validate_data/
    ├── README.md               # this file
    ├── requirements.txt        # Python packages
    ├── producer.py             # fake stream: Spark → batches → stdout
    ├── validator.py            # reads batches from stdin, validates with pandera
    └── trx_stream_schema.yml   # the data contract for one batch
```

| File | What it does |
|---|---|
| `producer.py` | Reads `detail/trx` with PySpark, keeps only `fold = 1` (Spark opens only the `fold=1/` folder: partition pruning), sorts by `client_id, event_time` so runs are reproducible, splits into batches and randomly corrupts the data type of one column in some batches. Writes the batches to **stdout** and a log of what it corrupted (the ground truth) to **stderr**. |
| `validator.py` | Reads the batches from **stdin**, converts each to pandas with nullable types (`Int32`, `Float32`), validates it with the schema, and prints `PASS` / `FAIL` with the reason. Exits with code `1` if any batch failed. |
| `trx_stream_schema.yml` | All 15 columns with the exact types Spark sends (`str`, `datetime64[us, UTC]`, `Float32`, `Int32`), nullability and value checks. `strict: true` rejects unknown columns; `coerce: false` reports wrong types instead of silently converting them. |

### Injected errors

| Error | Expected type | What the validator sees |
|---|---|---|
| `amount -> string` | `Float32` | `str` |
| `event_type -> float64` | `Int32` | `Float64` |
| `event_time -> string` | `datetime64[us, UTC]` | `str` |
| `client_id -> int64` | `str`, 64 hex chars | integer |
| `event_subtype -> int64` | `Int32` | `Int64` (still an integer, but 64-bit) |

## Environment setup

Tested on macOS (Apple Silicon) with Python 3.13.

### 1. Java 17 (needed by Spark)

```bash
brew install openjdk@17
```

Homebrew's `openjdk@17` is "keg-only", so add it to your shell once:

```bash
echo 'export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"' >> ~/.zshrc && echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
```

```bash
java -version
```

`producer.py` falls back to `/opt/homebrew/opt/openjdk@17/...` if `JAVA_HOME` is not set.

### 2. Python packages

```bash
pip install -r validate_data/requirements.txt
```

### 3. Data

Download and extract MBD-mini (about 3.4 GB download, about 3.6 GB extracted):

```bash
bash data/script.sh
```

The producer expects `data/mbd_mini/detail/trx/fold=1/` to exist.

## How to run

From the project root:

```bash
cd validate_data
```

```bash
python producer.py --seed 42 | python validator.py
```

The validator's results go to the terminal; the producer's log (which batches it corrupted) also appears in the terminal via stderr. To keep them apart, save the producer log to a file:

```bash
python producer.py --seed 42 2> producer.log | python validator.py
```

```bash
cat producer.log
```

### Example output (seed 42)

```
batch 0  PASS
batch 1  FAIL
    event_time: dtype('datetime64[us, UTC]') (e.g. 'str', 1 case(s))
batch 2  FAIL
    event_type: dtype('Int32') (e.g. 'Float64', 1 case(s))
batch 3  PASS
batch 4  PASS
batch 5  PASS
batch 6  FAIL
    client_id: dtype('string[pyarrow]') (e.g. 4532308, 1000 case(s))
    client_id: str_matches('^[0-9a-f]{64}$') (...)
batch 7  FAIL
    amount: dtype('Float32') (e.g. 'str', 1 case(s))
    amount: greater_than(0) (...)
batch 8  FAIL
    event_subtype: dtype('Int32') (e.g. 'Int64', 1 case(s))
batch 9  PASS

10 batches, 10,000 rows: 5 passed, 5 failed
```

Compare it with `producer.log`: every batch marked `CORRUPTED` should be `FAIL`, every `clean` batch should be `PASS`. A full run takes about 15–20 seconds, mostly Spark start-up.

### Options

`producer.py`:

| Option | Default | Meaning |
|---|---|---|
| `--fold` | `1` | Which fold to stream (0–4) |
| `--batches` | `10` | Number of batches |
| `--batch-size` | `1000` | Rows per batch |
| `--error-rate` | `0.3` | Probability that a batch gets a type error (`0` = all clean) |
| `--seed` | none | Random seed; the same seed corrupts the same batches |

`validator.py`:

| Option | Default | Meaning |
|---|---|---|
| `--schema` | `trx_stream_schema.yml` | Path to the pandera YAML schema |

Examples:

```bash
python producer.py --error-rate 0 | python validator.py
```

```bash
python producer.py --fold 3 --batches 50 --seed 7 | python validator.py
```

## How it works

1. **Read.** Spark reads `detail/trx`, filters `fold = 1` and takes the first `batches × batch-size` rows.
2. **Send.** Each batch is sent as its own Arrow IPC message with an 8-byte length in front. A single Arrow stream requires every batch to have the same types; sending them separately lets a corrupted batch carry different types. Arrow is used instead of CSV/JSON because it keeps the exact data types.
3. **Convert.** The validator turns each batch into pandas with nullable types. Without this, a batch with a missing `src_type11` would become `float64` and fail even though the data is fine.
4. **Validate.** `schema.validate(df, lazy=True)` runs all checks and collects every failure instead of stopping at the first one.
5. **Report.** One line per batch, a summary at the end, and exit code `1` if anything failed.

## Limitations

- Checks run **per batch**. Rules that span the whole table (unique keys, "profile columns constant per client", category hierarchies, fold matches `client_split`) need a separate full-table check.
- PySpark 4.3 warns that it does not fully support pandas 3. It did not affect these scripts; the validator hides the warning.
- `data/transaction_schema.yml` is an older draft and is **not** used here (it allows only 2-character `client_id`s and uses `coerce: true`).
