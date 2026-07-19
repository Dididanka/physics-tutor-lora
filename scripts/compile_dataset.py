#!/usr/bin/env python3
"""Canonicalize, deduplicate, and deterministically split curated SFT data."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from dataset_io import (
    canonicalize_record,
    read_jsonl,
    user_prompt_key,
    validation_errors,
    write_jsonl,
)


def split_count(total: int, fraction: float, minimum_remaining: int) -> int:
    """Choose a non-empty split while reserving records for later splits/train."""
    if fraction == 0:
        return 0
    return max(1, min(total - minimum_remaining, round(total * fraction)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="directory of source JSONL files")
    parser.add_argument("--output", type=Path, required=True, help="directory for train/valid/test JSONL")
    parser.add_argument("--seed", type=int, default=42, help="deterministic split seed")
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"input directory does not exist: {args.input}")
    if not 0 <= args.valid_fraction < 1 or not 0 <= args.test_fraction < 1:
        parser.error("split fractions must be in [0, 1)")
    if args.valid_fraction + args.test_fraction >= 1:
        parser.error("valid and test fractions together must be below 1")

    records: list[dict] = []
    seen_prompts: set[str] = set()
    rejected = 0
    source_files = sorted(args.input.glob("*.jsonl"))
    if not source_files:
        parser.error(f"no .jsonl files found in {args.input}")

    for source in source_files:
        try:
            source_records = list(read_jsonl(source))
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        for line_number, record in source_records:
            source_errors = validation_errors(record)
            if source_errors:
                rejected += 1
                print(
                    f"Skipping {source}:{line_number}: {'; '.join(source_errors)}",
                    file=sys.stderr,
                )
                continue
            normalized = canonicalize_record(record)
            key = user_prompt_key(normalized)
            if key in seen_prompts:
                rejected += 1
                print(f"Skipping {source}:{line_number}: duplicate user prompt", file=sys.stderr)
                continue
            seen_prompts.add(key)
            records.append(normalized)

    if len(records) < 3:
        print("Need at least three valid, unique records to create train/valid/test splits.", file=sys.stderr)
        return 1

    random.Random(args.seed).shuffle(records)
    test_size = split_count(len(records), args.test_fraction, minimum_remaining=2)
    valid_size = split_count(
        len(records) - test_size, args.valid_fraction, minimum_remaining=1
    )
    test_records = records[:test_size]
    valid_records = records[test_size : test_size + valid_size]
    train_records = records[test_size + valid_size :]

    paths_and_records = (
        (args.output / "train.jsonl", train_records),
        (args.output / "valid.jsonl", valid_records),
        (args.output / "test.jsonl", test_records),
    )
    for path, split_records in paths_and_records:
        write_jsonl(path, split_records)

    print(
        "Compiled "
        f"{len(records)} record(s): train={len(train_records)}, "
        f"valid={len(valid_records)}, test={len(test_records)}, rejected={rejected}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
