"""Logging helpers for fetch-agent tool loops."""

import json
from typing import Any

from eu_climate_policy_rag.core.logging_utils import get_logger

LOGGER = get_logger(__name__)


def preview_args(args: dict[str, Any]) -> dict[str, str]:
    """Return a compact string preview of tool arguments."""

    return {key: str(value)[:80] for key, value in args.items()}


def log_message_output(output: list[Any]) -> None:
    """Log text blocks from LLM message outputs."""

    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", []):
            text = getattr(block, "text", None)
            if text:
                LOGGER.info("LLM: %s", text.strip())


def log_tool_result(tool_name: str, result: str) -> None:
    """Log a compact summary of one fetch-tool result."""

    result_preview = json.loads(result)
    if tool_name == "get_page_snapshot":
        LOGGER.info(
            'Page "%s": %s links, %s doc links, %s download buttons',
            result_preview.get("title", ""),
            len(result_preview.get("links", [])),
            len(result_preview.get("document_links", [])),
            len(result_preview.get("download_buttons", [])),
        )
    elif tool_name in {"click_and_capture", "click_download_button"}:
        LOGGER.info(
            'Fetched "%s" (%s), content_id=%s',
            result_preview.get("title", ""),
            result_preview.get("format", "?"),
            result_preview.get("content_id", "?"),
        )
    elif tool_name == "convert_to_markdown":
        preselection = result_preview.get("preselection", {})
        LOGGER.info(
            "Converted %s chars, markdown_id=%s, preselection=%s",
            f"{result_preview.get('length', 0):,}",
            result_preview.get("markdown_id", "?"),
            preselection.get("reason", "?"),
        )
    elif tool_name == "save_content_to_file":
        if result_preview.get("rejected"):
            LOGGER.warning("Rejected save: %s", result_preview.get("reason", "?"))
            return
        LOGGER.info(
            "Saved to %s (%s bytes)",
            result_preview.get("path", "?"),
            f"{result_preview.get('bytes', 0):,}",
        )
    else:
        LOGGER.info("Tool result: %s", json.dumps(result_preview)[:140])
