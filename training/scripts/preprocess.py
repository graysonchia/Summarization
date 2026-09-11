"""Phase 3: unify and tokenize the three summarization datasets for FLAN-T5."""

from collections import Counter
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
TASK_PREFIX = "summarize: "
SEED = 42
DATASET_REPOS = {
    "cnn_dailymail": ("abisee/cnn_dailymail", "3.0.0"),
    "xsum": ("EdinburghNLP/xsum",),
    "samsum": ("knkarthick/samsum",),
}
FIELD_MAP = {
    "cnn_dailymail": {"article": "text", "highlights": "summary"},
    "xsum": {"document": "text", "summary": "summary"},
    "samsum": {"dialogue": "text", "summary": "summary"},
}
TRAIN_CAPS = {"cnn_dailymail": 15_000, "xsum": 15_000, "samsum": 10_000}
SPLITS = ("train", "validation", "test")
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "tokenized_dataset"


def load_unified_split(split):
    """Preserve source provenance while combining reproducibly selected rows."""
    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split!r}; expected one of {SPLITS}")
    parts = []
    for source, repo_args in DATASET_REPOS.items():
        print(f"Loading {source}/{split}...", flush=True)
        dataset = load_dataset(*repo_args, split=split)
        if split == "train":
            dataset = dataset.shuffle(seed=SEED).select(
                range(min(TRAIN_CAPS[source], len(dataset)))
            )
        mapping = FIELD_MAP[source]
        dataset = dataset.select_columns(list(mapping))
        dataset = dataset.rename_columns({
            original: common for original, common in mapping.items() if original != common
        })
        dataset = dataset.add_column("source", [source] * len(dataset))
        parts.append(dataset)
    return concatenate_datasets(parts).shuffle(seed=SEED)


def tokenize_batch(examples, tokenizer):
    """Encode prefixed inputs and targets; ignore target padding in the loss."""
    inputs = [TASK_PREFIX + text for text in examples["text"]]
    encoded = tokenizer(
        inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length"
    )
    targets = tokenizer(
        text_target=examples["summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )
    encoded["labels"] = [
        [token if token != tokenizer.pad_token_id else -100 for token in label]
        for label in targets["input_ids"]
    ]
    return encoded


def main():
    print(f"Loading tokenizer: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    unified = DatasetDict({split: load_unified_split(split) for split in SPLITS})
    print(f"\nUnified train total: {len(unified['train']):,}", flush=True)
    counts = Counter(unified["train"]["source"])
    for source in DATASET_REPOS:
        print(f"  {source}: {counts[source]:,}", flush=True)

    tokenized = unified.map(
        tokenize_batch,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["text", "summary", "source"],
        desc="Tokenizing",
    )
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(OUTPUT_DIR))
    print(f"\nSaved tokenized dataset to {OUTPUT_DIR}")
    for split in SPLITS:
        print(f"  {split}: {len(tokenized[split]):,} rows")


if __name__ == "__main__":
    main()
