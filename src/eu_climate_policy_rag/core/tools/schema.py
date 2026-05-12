"""Schema normalization for OpenAI Responses strict function tools."""

from collections.abc import Mapping
from copy import deepcopy

from eu_climate_policy_rag.core.tools.errors import SchemaGenerationError

STRIPPED_KEYS = {
    "default",
    "examples",
    "$schema",
}

UNSUPPORTED_KEYS = {
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
    normalized = _normalize_node(working, definitions, ref_stack=())
    if normalized.get("type") != "object":
        normalized["type"] = "object"
    return normalized


def _normalize_node(
    node: object,
    definitions: Mapping[str, object],
    *,
    ref_stack: tuple[str, ...],
) -> object:
    if isinstance(node, list):
        return [
            _normalize_node(item, definitions, ref_stack=ref_stack) for item in node
        ]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        ref = node["$ref"]
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.removeprefix("#/$defs/")
            if name in ref_stack:
                path = " -> ".join((*ref_stack, name))
                msg = f"OpenAI Responses strict schema does not support recursive $ref: {path}."
                raise SchemaGenerationError(msg, details={"ref": ref, "path": path})
            target = definitions.get(name)
            if target is not None:
                return _normalize_node(
                    target,
                    definitions,
                    ref_stack=(*ref_stack, name),
                )
            msg = f"OpenAI Responses strict schema contains unresolved $ref: {ref}."
            raise SchemaGenerationError(msg, details={"ref": ref})
        msg = f"OpenAI Responses strict schema does not support $ref: {ref}."
        raise SchemaGenerationError(msg, details={"ref": ref})

    normalized: dict[str, object] = {}
    for key, value in node.items():
        if key in STRIPPED_KEYS or key == "$defs":
            continue
        if key in UNSUPPORTED_KEYS:
            msg = (
                "OpenAI Responses strict schema does not support "
                f"JSON Schema keyword {key!r}."
            )
            raise SchemaGenerationError(msg, details={"keyword": key})
        normalized[key] = _normalize_node(
            value,
            definitions,
            ref_stack=ref_stack,
        )

    if normalized.get("type") == "object" or "properties" in normalized:
        additional_properties = normalized.get("additionalProperties")
        if additional_properties not in (None, False):
            msg = (
                "OpenAI Responses strict schema requires object schemas to use "
                "additionalProperties: false; open object/map schemas are not "
                "supported."
            )
            raise SchemaGenerationError(
                msg,
                details={"additionalProperties": additional_properties},
            )
        properties = normalized.get("properties")
        if not isinstance(properties, dict):
            properties = {}
            normalized["properties"] = properties
        normalized["additionalProperties"] = False
        normalized["required"] = list(properties)

    if normalized.get("type") == "array" and "items" not in normalized:
        normalized["items"] = {}

    return normalized
