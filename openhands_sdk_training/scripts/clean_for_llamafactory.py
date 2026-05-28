#!/usr/bin/env python3
"""Clean ADP ShareGPT JSONL files for LLaMA-Factory/Qwen VL processing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MEDIA_TAG_REPLACEMENTS = {
    "<image>": "&lt;image&gt;",
    "</image>": "&lt;/image&gt;",
    "<video>": "&lt;video&gt;",
    "</video>": "&lt;/video&gt;",
    "<audio>": "&lt;audio&gt;",
    "</audio>": "&lt;/audio&gt;",
}


def clean_text(value: str) -> tuple[str, bool]:
    changed = False
    for old, new in MEDIA_TAG_REPLACEMENTS.items():
        if old in value:
            value = value.replace(old, new)
            changed = True
    return value, changed


def clean_record(record: dict[str, Any]) -> tuple[dict[str, Any] | None, bool, str | None]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return None, False, "missing_or_empty_conversations"

    changed = False
    for message in conversations:
        if not isinstance(message, dict):
            return None, changed, "non_object_message"
        if not isinstance(message.get("from"), str):
            return None, changed, "missing_role"
        value = message.get("value")
        if not isinstance(value, str):
            return None, changed, "missing_content"
        new_value, text_changed = clean_text(value)
        if text_changed:
            message["value"] = new_value
            changed = True

    if "system" in record and record["system"] is not None and not isinstance(record["system"], str):
        record["system"] = str(record["system"])
        changed = True

    return record, changed, None


def clean_file(input_path: Path, output_path: Path) -> dict[str, int]:
    stats = {
        "read": 0,
        "written": 0,
        "json_errors": 0,
        "invalid_records": 0,
        "media_tag_rows_changed": 0,
    }

    with input_path.open("r", encoding="utf-8") as in_handle, output_path.open("w", encoding="utf-8") as out_handle:
        for line in in_handle:
            stats["read"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                continue

            if not isinstance(record, dict):
                stats["invalid_records"] += 1
                continue

            cleaned, changed, invalid_reason = clean_record(record)
            if invalid_reason is not None or cleaned is None:
                stats["invalid_records"] += 1
                continue

            if changed:
                stats["media_tag_rows_changed"] += 1
            out_handle.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            stats["written"] += 1

    return stats


def default_output(path: Path) -> Path:
    return path.with_name(path.stem + "_clean_mm_safe" + path.suffix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--train-out", type=Path)
    parser.add_argument("--eval-out", type=Path)
    args = parser.parse_args()

    train = args.train.expanduser()
    eval_path = args.eval.expanduser()
    train_out = (args.train_out.expanduser() if args.train_out else default_output(train))
    eval_out = (args.eval_out.expanduser() if args.eval_out else default_output(eval_path))

    train_stats = clean_file(train, train_out)
    eval_stats = clean_file(eval_path, eval_out)

    print(json.dumps({"train": str(train_out), "stats": train_stats}, indent=2))
    print(json.dumps({"eval": str(eval_out), "stats": eval_stats}, indent=2))


if __name__ == "__main__":
    main()

