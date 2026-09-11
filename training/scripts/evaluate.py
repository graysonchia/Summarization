"""Evaluate unmodified FLAN-T5 on the Phase 3 test set, before fine-tuning."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

# This filename otherwise shadows the third-party `evaluate` package when run
# as `python scripts/evaluate.py` from training/.
if __package__ in (None, ""):
    sys.path[0] = str(Path(__file__).resolve().parents[2])

import evaluate
from datasets import load_from_disk
from nltk.tokenize import PunktSentenceTokenizer
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from training.scripts.preprocess import (
    DATASET_REPOS, MAX_TARGET_LENGTH, MODEL_NAME, OUTPUT_DIR, SEED,
    load_unified_split, tokenize_batch,
)


ROUGE_KEYS = ("rouge1", "rouge2", "rougeL", "rougeLsum")
SENTENCE_TOKENIZER = PunktSentenceTokenizer()


def select_indices(sources, sample_size):
    """Proportional stratified sample with at least three rows per source."""
    if sample_size is None or sample_size >= len(sources):
        return list(range(len(sources)))
    if sample_size < 9:
        raise ValueError("--sample-size must be at least 9 (three examples per source)")
    groups = {source: [] for source in DATASET_REPOS}
    for index, source in enumerate(sources):
        groups[source].append(index)
    if any(len(group) < 3 for group in groups.values()):
        raise ValueError("Each source needs at least three test examples")
    quotas = {source: 3 for source in groups}
    for _ in range(sample_size - sum(quotas.values())):
        source = max(
            (source for source in groups if quotas[source] < len(groups[source])),
            key=lambda source: sample_size * len(groups[source]) / len(sources) - quotas[source],
        )
        quotas[source] += 1
    rng = random.Random(SEED)
    return sorted(index for source, group in groups.items()
                  for index in rng.sample(group, quotas[source]))


def verify_alignment(tokenized, raw, tokenizer):
    """Compare every selected row, including targets, before trusting provenance."""
    for start in tqdm(range(0, len(raw), 256), desc="Checking source alignment"):
        stop = min(start + 256, len(raw))
        rebuilt = tokenize_batch(raw[start:stop], tokenizer)
        saved = tokenized[start:stop]
        for key in ("input_ids", "attention_mask", "labels"):
            if rebuilt[key] != saved[key]:
                raise ValueError(
                    f"Source alignment mismatch for {key} in rows {start}:{stop}. "
                    "Rebuild Phase 3 data using the current datasets/tokenizer before evaluating."
                )


def compute_scores(metric, predictions, references):
    # ROUGE-Lsum uses newline-separated sentences. This resource-free Punkt
    # splitter is shared by all groups; use the same convention in Phase 7.
    def sentences(text):
        return "\n".join(SENTENCE_TOKENIZER.tokenize(text.strip()))

    np.random.seed(SEED)  # evaluate's bootstrap aggregation uses NumPy's RNG.
    scores = metric.compute(
        predictions=[sentences(text) for text in predictions],
        references=[sentences(text) for text in references],
        use_stemmer=True,
    )
    return {key: float(scores[key]) for key in ROUGE_KEYS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Proportional stratified sample, minimum 9; default: full test set")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR.parent)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.sample_size is not None and args.sample_size < 9):
        parser.error("batch size must be positive; sample size must be at least 9")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cpu":
        print("WARNING: Full-test generation on CPU will be slow. "
              "Use --sample-size 500 for an initial check or run on a GPU.", flush=True)
        # Avoid excessive thread contention on machines with many logical cores.
        torch.set_num_threads(min(8, torch.get_num_threads()))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test = load_from_disk(str(OUTPUT_DIR / "test"))
    raw = load_unified_split("test")
    if len(test) != len(raw):
        raise ValueError("Saved and rebuilt test split sizes differ; regenerate Phase 3 data")
    indices = select_indices(raw["source"], args.sample_size)
    test, raw = test.select(indices), raw.select(indices)
    sources = list(raw["source"])
    verify_alignment(test, raw, tokenizer)
    print(f"Evaluating {len(test):,} examples: {dict(Counter(sources))}", flush=True)

    metric = evaluate.load("rouge")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    predictions, references, qualitative = [], [], []
    qualitative_counts = Counter()
    with torch.inference_mode():
        for start in tqdm(range(0, len(test), args.batch_size), desc="Generating summaries"):
            batch = test[start:start + args.batch_size]
            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
            # Trim only batch-wide padding; preserve the saved input tokens exactly.
            width = int(attention_mask.sum(dim=1).max().item())
            generated = model.generate(
                input_ids=input_ids[:, :width], attention_mask=attention_mask[:, :width],
                num_beams=4, max_length=MAX_TARGET_LENGTH, do_sample=False,
            )
            decoded_inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            decoded_predictions = tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True)
            labels = [[tokenizer.pad_token_id if token == -100 else token for token in row]
                      for row in batch["labels"]]
            decoded_references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            predictions.extend(decoded_predictions)
            references.extend(decoded_references)
            for offset, (text, prediction, reference) in enumerate(
                zip(decoded_inputs, decoded_predictions, decoded_references)
            ):
                source = sources[start + offset]
                if qualitative_counts[source] < 3:
                    qualitative.append({
                        "source": source, "test_index": indices[start + offset],
                        "input_excerpt": text[:1000], "generated_summary": prediction,
                        "reference_summary": reference,
                    })
                    qualitative_counts[source] += 1

    pooled = compute_scores(metric, predictions, references)
    by_source = {}
    for source in DATASET_REPOS:
        selected = [i for i, label in enumerate(sources) if label == source]
        by_source[source] = compute_scores(
            metric, [predictions[i] for i in selected], [references[i] for i in selected]
        )
    print("\nROUGE scores (0-1 scale)")
    print(f"{'Source':<18} {'Count':>7}" + "".join(f"{key:>12}" for key in ROUGE_KEYS))
    for source, scores in {"pooled": pooled, **by_source}.items():
        count = len(test) if source == "pooled" else sources.count(source)
        print(f"{source:<18} {count:>7}" + "".join(f"{scores[key]:>12.4f}" for key in ROUGE_KEYS))

    results = {
        "pooled": pooled, "by_source": by_source,
        "model": MODEL_NAME, "num_examples": len(test),
        "evaluation_config": {
            "seed": SEED, "test_indices": indices, "source_counts": dict(Counter(sources)),
            "sample_size_requested": args.sample_size, "batch_size": args.batch_size,
            "num_beams": 4, "max_length": MAX_TARGET_LENGTH,
            "use_stemmer": True, "sentence_splitter": "nltk.PunktSentenceTokenizer (untrained)",
            "references": "decoded Phase 3 labels, truncated to 128 tokens",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in (
        ("baseline_results.json", results),
        ("baseline_qualitative_samples.json", qualitative),
    ):
        path = args.output_dir / filename
        path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
