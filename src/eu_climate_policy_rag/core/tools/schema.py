"""Schema normalization for OpenAI Responses strict function tools."""

from collections.abc import Mapping
from copy import deepcopy

UNSUPPORTED_KEYS = {
    "$schema",
    "default",
    "examples",
    "allOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "dependentRequired",
    "dependentSchemas",
    "patternProperties",
}


def normalize_openai_schema(schema: Mapping[str, object]) -> dict[str, object]:
    """Normalize a JSON Schema-like mapping for OpenAI strict function tools."""

    working = deepcopy(dict(schema))
    definitions = working.get("$defs")
    if not isinstance(definitions, dict):
        definitions = {}
    normalized = _normalize_node(working, definitions)
    if normalized.get("type") != "object":
        normalized["type"] = "object"
    return normalized


def _normalize_node(
    node: object,
    definitions: Mapping[str, object],
) -> object:
    if isinstance(node, list):
        return [_normalize_node(item, definitions) for item in node]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        ref = node["$ref"]
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.removeprefix("#/$defs/")
            target = definitions.get(name)
            if target is not None:
                return _normalize_node(target, definitions)

    normalized: dict[str, object] = {}
    for key, value in node.items():
        if key in UNSUPPORTED_KEYS or key == "$defs":
            continue
        normalized[key] = _normalize_node(value, definitions)

    if normalized.get("type") == "object" or "properties" in normalized:
        properties = normalized.get("properties")
        if not isinstance(properties, dict):
            properties = {}
            normalized["properties"] = properties
        normalized["additionalProperties"] = False
        normalized["required"] = list(properties)

    if normalized.get("type") == "array" and "items" not in normalized:
        normalized["items"] = {}

    return normalized
