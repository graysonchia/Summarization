# Text Summarizer

Phase 1 provides the project structure and development environments for a future
text-summarization training pipeline, FastAPI backend, and React frontend. Model
training and inference logic will be added in later phases.

## Project structure

- `training/` contains notebook and script placeholders for the training pipeline.
- `backend/` contains a minimal FastAPI application.
- `frontend/` contains a minimal Vite, React, TypeScript, and Tailwind application.

## Training environment

```powershell
cd training
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For Kaggle, upload `training/requirements.txt` and install it from a notebook cell:

```python
!pip install -r requirements.txt
```

## Data acquisition (Phase 2)

From `training/`, with the training environment active, run:

```powershell
python scripts/prepare_data.py
```

This downloads CNN/DailyMail 3.0.0, XSum, and SAMSum into the default Hugging Face
cache (or `HF_HOME`/`HF_DATASETS_CACHE` if configured). It prints split sizes,
columns, one raw training example per dataset, and word-count statistics for the
first up to 1,000 training examples. These are sampled counts, not full-dataset
maxima or tokenizer lengths. No preprocessing is performed.

On success, the combined report is saved to `training/data/dataset_stats.csv`,
regardless of the working directory. If a source fails, the script still checks
the others, prints the exact errors, exits unsuccessfully, and does not write a
combined report. Any CSV from a previous run is left unchanged.

## Unification and tokenization (Phase 3)

From `training/`, with the training environment active, run:

```powershell
python scripts/preprocess.py
```

The script samples 15,000 CNN/DailyMail, 15,000 XSum, and 10,000 SAMSum training
rows with seed 42 and uses the complete validation and test splits. It prints the
training source counts before tokenization. FLAN-T5 inputs receive the
`summarize: ` prefix and are padded/truncated to 512 tokens; targets are padded/
truncated to 128 tokens, with padding replaced by `-100` to ignore it in the loss.

The resulting `DatasetDict` is saved to `training/data/tokenized_dataset/` and
can be restored with `datasets.load_from_disk`. Raw text, summaries, and source
names are removed in the tokenization map. For later per-source evaluation,
rebuild the unified split with `load_unified_split` using the same source data
and seed to recover source labels in the corresponding row order.

## Baseline evaluation (Phase 5)

From `training/`, with the training environment active:

```powershell
python scripts/evaluate.py --sample-size 500
# Full test set (23,643 examples):
python scripts/evaluate.py
```

The script loads the unmodified `google/flan-t5-base` model, uses CUDA when
available, and otherwise warns about CPU runtime. `--batch-size` defaults to 16;
reduce it if memory is limited. For a minimal CPU smoke test, use
`--sample-size 9 --batch-size 3`. Samples are reproducible and proportional to
source sizes, subject to a minimum of three examples per source. Very small
samples therefore do not represent the full test distribution.

Source labels are recovered through Phase 3's `load_unified_split("test")`.
Every selected row is retokenized and compared against the saved input IDs,
attention mask, and labels before its source is trusted. A mismatch stops the
run. Generation uses four beams, a maximum length of 128, and the saved inputs.
References are decoded from the saved labels, so they retain Phase 3's target
truncation. Label padding (`-100`) is restored before decoding.

ROUGE scores use the `evaluate` ROUGE metric with stemming and a resource-free
NLTK Punkt sentence splitter for newline-separated ROUGE-Lsum scoring. Scores
are on a 0-1 scale. Preserve these settings and the recorded test indices for
the Phase 7 comparison.

Results are written to `training/data/baseline_results.json`, including pooled
and per-source scores, sample indices, counts, and generation/scoring settings.
`training/data/baseline_qualitative_samples.json` contains nine examples (three
per source). `--output-dir` redirects both reports; reruns replace reports in
the selected directory. Check `num_examples` before treating a smoke-test report
as a full baseline. Full evaluation is best run on a GPU if CPU timing is too slow.

## Kaggle fine-tuning and full comparison (Phases 6–7)

Import `training/notebooks/02_finetune.ipynb` into Kaggle. Enable Internet and a
T4 or P100 GPU, then run cells in order. The notebook rebuilds the same Phase 3
data directly from the Hub, evaluates the unmodified model on the full test set,
fine-tunes for three epochs, and evaluates the best validation checkpoint on the
identical test rows. No repository or dataset upload is needed.

Outputs are under `/kaggle/working/training/`: the authoritative
`data/baseline_results_full.json`, `data/finetuned_results_full.json`, per-source
comparisons, qualitative examples, training curves, provenance manifest, and
`checkpoints/flan-t5-summarizer/best/` with the model and tokenizer. The notebook
also creates a best-model ZIP for downloading. Save a Kaggle version with outputs
retained and download them from its Output panel before ending the session.

Full validation generation every 500 steps can dominate runtime; the notebook
keeps the requested full validation and test sets. Batch size and FP16 settings
are exposed with OOM and non-finite-loss recovery notes. GPU execution and model
improvement must be verified in Kaggle; the notebook contains no fabricated scores.

Offline notebook checks (from the repository root):

```powershell
.\training\venv\Scripts\python.exe -m unittest discover -s training/tests -v
```

## Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

The health endpoint is available at `http://127.0.0.1:8000/health`.

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

Vite prints the local development URL when it starts.
