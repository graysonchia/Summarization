"""Phase 2: acquire and inspect CNN/DailyMail, XSum, and SAMSum."""

from pathlib import Path

from datasets import load_dataset
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIELD_MAP = {
    "cnn_dailymail": ("article", "highlights"),
    "xsum": ("document", "summary"),
    "samsum": ("dialogue", "summary"),
}


def load_all_datasets():
    """Try every requested source and retain exact errors for failed loads."""
    sources = {
        "cnn_dailymail": ("abisee/cnn_dailymail", "3.0.0"),
        "xsum": ("EdinburghNLP/xsum",),
        "samsum": ("knkarthick/samsum",),
    }
    loaded = {}
    errors = {}
    for name, args in sources.items():
        print(f"Loading {args}...", flush=True)
        try:
            loaded[name] = load_dataset(*args)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            print(f"FAILED {name}: {errors[name]}", flush=True)
    return loaded, errors


def inspect_dataset(name, ds):
    print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")
    for split, records in ds.items():
        print(f"  {split}: {len(records)} examples")
        print(f"  {split} columns: {records.column_names}")
    if "train" not in ds or not len(ds["train"]):
        raise ValueError(f"{name}: missing or empty training split")
    print(f"  Sample: {ds['train'][0]}")
    for split in ("train", "validation", "test"):
        if split not in ds or not len(ds[split]):
            raise ValueError(f"{name}: missing or empty {split} split")
        missing = set(FIELD_MAP[name]) - set(ds[split].column_names)
        if missing:
            raise ValueError(f"{name}/{split}: missing columns {sorted(missing)}")


def compute_length_stats(name, ds, text_col, summary_col, sample_size=1000):
    if sample_size <= 0 or not len(ds["train"]):
        raise ValueError("Length statistics require a positive sample size and nonempty train split")
    sample = ds["train"].select(range(min(sample_size, len(ds["train"]))))
    lengths = {}
    for col in (text_col, summary_col):
        values = sample[col]
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"{name}: sampled column {col!r} contains blank or non-string values")
        lengths[col] = [len(value.split()) for value in values]
    return pd.DataFrame({
        "dataset": [name],
        "avg_text_words": [sum(lengths[text_col]) / len(sample)],
        "avg_summary_words": [sum(lengths[summary_col]) / len(sample)],
        "max_text_words": [max(lengths[text_col])],
        "max_summary_words": [max(lengths[summary_col])],
    })


def main():
    datasets, errors = load_all_datasets()
    all_stats = []
    for name, ds in datasets.items():
        try:
            inspect_dataset(name, ds)
            all_stats.append(compute_length_stats(name, ds, *FIELD_MAP[name]))
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            print(f"FAILED {name}: {errors[name]}")

    if all_stats:
        stats_df = pd.concat(all_stats, ignore_index=True)
        print("\nLength statistics (word counts, first up to 1,000 training examples):")
        print(stats_df.to_string(index=False))
    if errors:
        print("\nAcquisition/inspection incomplete; no combined CSV written this run.")
        for name, error in errors.items():
            print(f"  {name}: {error}")
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / "dataset_stats.csv"
    stats_df.to_csv(output, index=False)
    print(f"\nSaved stats to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
