#!/usr/bin/env python3
"""Validate JSONL SFT data before an MLX-LM training run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dataset_io import read_jsonl, validation_errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="JSONL file to validate")
    parser.add_argument(
        "--require-canonical-persona",
        action="store_true",
        help="require the exact project system persona in every record",
    )
    args = parser.parse_args()

    if not args.dataset.is_file():
        parser.error(f"dataset does not exist: {args.dataset}")

    records = 0
    errors = 0
    try:
        for line_number, record in read_jsonl(args.dataset):
            records += 1
            for error in validation_errors(
                record, require_canonical_persona=args.require_canonical_persona
            ):
                errors += 1
                print(f"{args.dataset}:{line_number}: {error}", file=sys.stderr)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    if records == 0:
        print(f"{args.dataset}: no records found", file=sys.stderr)
        return 1
    if errors:
        print(f"Validation failed: {errors} error(s) across {records} record(s).", file=sys.stderr)
        return 1

    print(f"Validated {records} record(s): {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
