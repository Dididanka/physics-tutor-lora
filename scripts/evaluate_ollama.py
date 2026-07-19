#!/usr/bin/env python3
"""Run held-out rubric prompts against a local Ollama model and record answers."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def read_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                case = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(case, dict) or not isinstance(case.get("messages"), list):
                raise ValueError(f"{path}:{line_number}: expected object with a messages list")
            if not isinstance(case.get("id"), str) or not isinstance(case.get("rubric"), list):
                raise ValueError(f"{path}:{line_number}: expected string id and list rubric")
            cases.append(case)
    if not cases:
        raise ValueError(f"{path}: no evaluation cases found")
    return cases


def call_ollama(
    host: str, model: str, messages: list[dict], timeout: int, think: bool
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": {"temperature": 0},
        }
    ).encode("utf-8")
    request = Request(
        f"{host.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {host}: {exc.reason}") from exc

    content = body.get("message", {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Ollama response has no assistant text: {body}")
    return content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="locally available Ollama model name")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/seed/test_prompts.jsonl"),
        help="held-out prompt/rubric JSONL",
    )
    parser.add_argument("--output", type=Path, required=True, help="result JSONL path")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--think",
        action="store_true",
        help="enable model thinking/reasoning mode (off by default for repeatable timing)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="run only the first N cases; 0 means run every case",
    )
    args = parser.parse_args()

    try:
        cases = read_cases(args.cases)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.max_cases < 0:
        parser.error("--max-cases must be zero or positive")
    if args.max_cases:
        cases = cases[: args.max_cases]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluated_at = datetime.now(UTC).isoformat()
    try:
        with args.output.open("w", encoding="utf-8") as handle:
            for index, case in enumerate(cases, start=1):
                answer = call_ollama(
                    args.host, args.model, case["messages"], args.timeout, args.think
                )
                result = {
                    "id": case["id"],
                    "model": args.model,
                    "evaluated_at": evaluated_at,
                    "messages": case["messages"],
                    "rubric": case["rubric"],
                    "answer": answer,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"[{index}/{len(cases)}] {case['id']}")
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Wrote {len(cases)} result(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
