"""Tests for the deterministic pure-SFT physics corpus builder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_physics_sft_corpus import generate  # noqa: E402
from dataset_io import validation_errors  # noqa: E402


class PhysicsSftBuilderTests(unittest.TestCase):
    def test_balanced_smoke_corpus_is_unique_and_valid(self) -> None:
        records, categories = generate(500, 20260719)
        self.assertEqual(len(records), 500)
        self.assertEqual(sum(categories.values()), 500)
        prompts = [record["messages"][1]["content"].casefold() for record in records]
        self.assertEqual(len(prompts), len(set(prompts)))
        for record in records:
            self.assertEqual(validation_errors(record, require_canonical_persona=True), [])


if __name__ == "__main__":
    unittest.main()
