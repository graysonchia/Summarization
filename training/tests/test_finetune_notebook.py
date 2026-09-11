"""Offline checks for the self-contained Kaggle notebook; no model training."""

import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from datasets import Dataset, concatenate_datasets
import numpy as np
from nltk.tokenize import PunktSentenceTokenizer

from training.scripts import preprocess


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/02_finetune.ipynb"


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, texts=None, *, text_target=None, max_length, **kwargs):
        rows = texts if text_target is None else text_target
        ids = [[ord(char) for char in text][:max_length] for text in rows]
        masks = [[1] * len(row) + [0] * (max_length - len(row)) for row in ids]
        ids = [row + [0] * (max_length - len(row)) for row in ids]
        return {"input_ids": ids, "attention_mask": masks}


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.ns = {name: getattr(preprocess, name) for name in (
            "SEED", "SPLITS", "DATASET_REPOS", "FIELD_MAP", "TASK_PREFIX",
            "MAX_INPUT_LENGTH", "MAX_TARGET_LENGTH",
        )}
        cls.ns.update(tokenizer=TinyTokenizer(), concatenate_datasets=concatenate_datasets,
                      TRAIN_CAPS={"cnn_dailymail": 3, "xsum": 3, "samsum": 2})
        for cell in cls.nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            # IPython's single installation magic is intentionally not Python syntax.
            source = "\n".join(line for line in source.splitlines() if not line.startswith("%pip"))
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), cls.ns)

    def test_notebook_structure(self):
        self.assertEqual(self.nb["nbformat"], 4)
        ids = [cell["id"] for cell in self.nb["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in self.nb["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_unification_matches_phase_three_on_every_split(self):
        def load_fixture(repo, *args, split):
            name = next(name for name, spec in preprocess.DATASET_REPOS.items() if spec[0] == repo)
            fields = list(preprocess.FIELD_MAP[name])
            return Dataset.from_dict({fields[0]: [f"{name}-{split}-{i}" for i in range(5)],
                                      fields[1]: [f"summary {i}" for i in range(5)],
                                      "id": list(range(5))})

        self.ns["load_dataset"] = load_fixture
        with patch.object(preprocess, "load_dataset", load_fixture), patch.object(
            preprocess, "TRAIN_CAPS", self.ns["TRAIN_CAPS"]
        ):
            for split in preprocess.SPLITS:
                actual = self.ns["build_unified_dataset"](split)
                expected = preprocess.load_unified_split(split)
                self.assertEqual(actual.to_dict(), expected.to_dict())
                self.assertEqual(len(actual), 8 if split == "train" else 15)
                self.assertEqual(set(actual.column_names), {"text", "summary", "source"})

    def test_tokenization_matches_phase_three(self):
        examples = {"text": ["short input", "x" * 700], "summary": ["target", "y" * 200]}
        result = self.ns["tokenize_batch"](examples)
        self.assertEqual(result, preprocess.tokenize_batch(examples, self.ns["tokenizer"]))
        self.assertEqual(result["labels"][0][-1], -100)
        self.assertEqual(len(result["input_ids"][1]), 512)
        self.assertEqual(len(result["labels"][1]), 128)

    def test_rouge_convention(self):
        class Metric:
            def compute(self, **kwargs):
                self.arguments = kwargs
                return dict.fromkeys(("rouge1", "rouge2", "rougeL", "rougeLsum"), 0.5)
        metric = Metric()
        self.ns.update(metric=metric, np=np, sentence_tokenizer=PunktSentenceTokenizer(),
                       ROUGE_KEYS=("rouge1", "rouge2", "rougeL", "rougeLsum"))
        scores = self.ns["compute_scores"](["First sentence. Second sentence."], ["A reference."])
        self.assertTrue(metric.arguments["use_stemmer"])
        self.assertIn("\n", metric.arguments["predictions"][0])
        self.assertEqual(scores["rougeL"], 0.5)


if __name__ == "__main__":
    unittest.main()
