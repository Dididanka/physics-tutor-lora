"""Small, dependency-free helpers for chat-format fine-tuning data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERSONA_PATH = PROJECT_ROOT / "prompts" / "physics_einstein_inspired_system.txt"
ALLOWED_ROLES = {"system", "user", "assistant"}


def canonical_persona() -> str:
    """Return the exact system prompt used in every training record."""
    return PERSONA_PATH.read_text(encoding="utf-8").strip()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield non-blank JSON objects with one-based source line numbers."""
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each line must be a JSON object")
            yield line_number, record


def validation_errors(
    record: dict[str, Any], *, require_canonical_persona: bool = False
) -> list[str]:
    """Return every schema error in a single SFT record."""
    errors: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return ["'messages' must be a list with at least two messages"]

    system_messages: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"message {index} must be an object")
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            errors.append(f"message {index} has unsupported role {role!r}")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"message {index} must have non-empty string content")
        if role == "system" and isinstance(content, str):
            system_messages.append(content.strip())

    first_role = messages[0].get("role") if isinstance(messages[0], dict) else None
    final_role = messages[-1].get("role") if isinstance(messages[-1], dict) else None
    if first_role != "system":
        errors.append("the first message must have role 'system'")
    if final_role != "assistant":
        errors.append("the final message must have role 'assistant'")
    if len(system_messages) != 1:
        errors.append("exactly one system message is required")
    elif require_canonical_persona and system_messages[0] != canonical_persona():
        errors.append("system message differs from the canonical physics persona")

    return errors


def canonicalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Replace any source persona with the project persona while preserving dialogue."""
    messages = record["messages"]
    non_system_messages = [message for message in messages if message.get("role") != "system"]
    return {"messages": [{"role": "system", "content": canonical_persona()}, *non_system_messages]}


def user_prompt_key(record: dict[str, Any]) -> str:
    """Stable key for detecting duplicate prompt/answer tasks."""
    user_content = [
        message["content"].strip()
        for message in record["messages"]
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    ]
    return "\n".join(user_content).casefold()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write compact UTF-8 JSONL and return record count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count
