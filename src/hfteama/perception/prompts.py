"""Prompts for the decomposer (categories) and the step check. Zero-shot: fixed answer format,
one in-context example, validator with retry. The VLM only references object and bin ids; the
positions come from SAM-6D, never from the model."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

SYSTEM = (
    "You are the vision module of a robot sorting cell. You look at a top-down photo of a table "
    "and decide how the objects should be sorted into bins. You do not control the robots. "
    "Answer with one JSON object only, no other text."
)

SORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["categories", "assignments"],
    "properties": {
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "rule", "bin"],
                "properties": {
                    "name": {"type": "string"},
                    "rule": {"type": "string"},
                    "bin": {"type": "string"},
                },
            },
        },
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["object", "category"],
                "properties": {"object": {"type": "string"}, "category": {"type": "string"}},
            },
        },
    },
}

_EXAMPLE = {
    "categories": [
        {"name": "red", "rule": "red objects", "bin": "bin_1"},
        {"name": "blue", "rule": "blue objects", "bin": "bin_2"},
    ],
    "assignments": [
        {"object": "obj_01", "category": "red"},
        {"object": "obj_02", "category": "blue"},
        {"object": "obj_03", "category": "red"},
    ],
}

CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["done", "confidence"],
    "properties": {
        "done": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
}


def sort_prompt(
    objects: Iterable[tuple[str, str, tuple[float, float] | None]],
    bins: Iterable[tuple[str, str]],
    fixed: dict[str, tuple[str, str]] | None = None,
    errors: list[str] | None = None,
) -> str:
    """objects: (id, description, pixel); bins: (id, label); fixed: category -> (rule, bin)."""
    bins = list(bins)
    lines = ["Objects on the table (each one is labelled with its id in the image):"]
    for oid, model, pixel in objects:
        where = f", at pixel ({pixel[0]:.0f}, {pixel[1]:.0f})" if pixel else ""
        kind = f" ({model})" if model else ""
        lines.append(f"- {oid}{kind}{where}")
    lines.append("")
    lines.append("Bins:")
    lines += [f"- {bid}: {label}" for bid, label in bins]
    lines.append("")
    lines.append(
        "Task: sort these objects. Decide the categories yourself from what you see "
        "(for example colour, shape or kind of object). "
        f"Use at most {len(bins)} categories. Give each category its own bin; "
        "no bin may be used by two categories. Put every listed object in exactly one category."
    )
    if fixed:
        lines.append("These categories already exist. Keep their names and bins unchanged:")
        lines += [f"- {name}: {rule} -> {b}" for name, (rule, b) in fixed.items()]
        lines.append("Add a new category only if an object fits none of them and a bin is free.")
    lines.append("")
    lines.append("Answer with JSON exactly in this form:")
    lines.append(
        '{"categories": [{"name": "<short name>", "rule": "<what belongs here>", '
        '"bin": "<bin id>"}], "assignments": [{"object": "<object id>", '
        '"category": "<category name>"}]}'
    )
    lines.append("Example answer for three objects and two bins:")
    lines.append(json.dumps(_EXAMPLE))
    if errors:
        lines.append("")
        lines.append("Your previous answer was rejected:")
        lines += [f"- {e}" for e in errors]
        lines.append("Answer again with corrected JSON.")
    return "\n".join(lines)


def check_prompt(question: str, errors: list[str] | None = None) -> str:
    lines = [
        "Look at the table in this photo.",
        f"Question: {question}",
        'Answer with JSON exactly in this form: {"done": true or false, '
        '"confidence": <number from 0 to 1>, "reason": "<a few words>"}',
    ]
    if errors:
        lines.append("Your previous answer was rejected: " + "; ".join(errors))
    return "\n".join(lines)


POINT_SYSTEM = (
    "You are the vision module of a robot sorting cell. You look at a top-down photo of a table "
    "and point at objects precisely. Answer with one JSON object only, no other text."
)


def point_prompt(errors: list[str] | None = None) -> str:
    lines = [
        "Point to every separate object lying on the table that a robot arm could pick up and "
        "sort. Do not point at bins, trays, boxes used as containers, robot arms, cables, "
        "markers or the table itself. Give exactly one point per object, at its centre.",
        "For an object that is longer than it is wide (or a square block), also give "
        '"axis_2d": two points at the middle of its two opposite short ends, so the line '
        "between them runs along the object. Leave axis_2d out for round objects.",
        'Answer with JSON exactly in this form: {"objects": [{"label": "<short description, '
        'e.g. red cube>", "point_2d": [x, y], "axis_2d": [[x1, y1], [x2, y2]]}]}',
        "All x and y are relative coordinates from 0 to 1000: [0, 0] is the top-left corner of "
        "the image and [1000, 1000] the bottom-right corner.",
    ]
    if errors:
        lines.append("Your previous answer was rejected: " + "; ".join(errors))
    return "\n".join(lines)
