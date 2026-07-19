"""Regression tests for data validation and deterministic splitting."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from dataset_io import canonical_persona, read_jsonl, validation_errors  # noqa: E402

import evaluate_ollama  # noqa: E402


class DatasetPipelineTests(unittest.TestCase):
    def test_seed_examples_match_the_canonical_persona(self) -> None:
        for name, expected_count in (("train.jsonl", 12), ("valid.jsonl", 4)):
            records = list(read_jsonl(PROJECT_ROOT / "data" / "seed" / name))
            self.assertEqual(len(records), expected_count)
            for _, record in records:
                self.assertEqual(
                    validation_errors(record, require_canonical_persona=True), [], name
                )

    def test_validator_rejects_an_answer_without_a_system_message(self) -> None:
        record = {
            "messages": [
                {"role": "user", "content": "Explain force."},
                {"role": "assistant", "content": "Force changes momentum."},
            ]
        }
        errors = validation_errors(record)
        self.assertIn("the first message must have role 'system'", errors)
        self.assertIn("exactly one system message is required", errors)

    def test_compiler_canonicalizes_and_makes_non_overlapping_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "raw"
            output = root / "processed"
            source.mkdir()
            source_records = []
            for index in range(5):
                source_records.append(
                    {
                        "messages": [
                            {"role": "system", "content": "temporary source persona"},
                            {"role": "user", "content": f"Question {index}"},
                            {"role": "assistant", "content": f"Answer {index}"},
                        ]
                    }
                )
            (source / "examples.jsonl").write_text(
                "\n".join(json.dumps(record) for record in source_records) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "compile_dataset.py"),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--seed",
                    "42",
                    "--valid-fraction",
                    "0.2",
                    "--test-fraction",
                    "0.2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            seen_prompts: set[str] = set()
            split_sizes = []
            for split in ("train", "valid", "test"):
                records = list(read_jsonl(output / f"{split}.jsonl"))
                split_sizes.append(len(records))
                for _, record in records:
                    self.assertEqual(record["messages"][0]["content"], canonical_persona())
                    prompt = record["messages"][1]["content"]
                    self.assertNotIn(prompt, seen_prompts)
                    seen_prompts.add(prompt)
            self.assertEqual(split_sizes, [3, 1, 1])

    def test_evaluator_request_can_disable_thinking(self) -> None:
        captured: dict = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def read(self) -> bytes:
                return b'{"message":{"content":"A clear answer."}}'

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        original = evaluate_ollama.urlopen
        evaluate_ollama.urlopen = fake_urlopen
        try:
            answer = evaluate_ollama.call_ollama(
                "http://example.invalid", "test-model", [], 12, think=False
            )
        finally:
            evaluate_ollama.urlopen = original

        self.assertEqual(answer, "A clear answer.")
        self.assertFalse(captured["payload"]["think"])
        self.assertEqual(captured["timeout"], 12)


if __name__ == "__main__":
    unittest.main()
