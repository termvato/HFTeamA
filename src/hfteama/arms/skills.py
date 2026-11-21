"""Skill registry: maps the `skill` name in a task graph to code that runs it on one arm.

A skill is a function (arm, task, poses) -> Outcome, where `arm` is a backend-specific handle
(real, MuJoCo or fake) exposing the same primitives. Generic skills (pick, place, pick_place,
inspect, park) live in the core; a task folder can add its own with the decorator:

    from hfteama.arms.skills import skill

    @skill("insert_peg")
    def insert_peg(arm, task, poses): ...
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

SkillFn = Callable[..., Any]
REGISTRY: dict[str, SkillFn] = {}


def skill(name: str) -> Callable[[SkillFn], SkillFn]:
    """Register a skill under `name`. Duplicate names are an error (two tasks clashing)."""

    def register(fn: SkillFn) -> SkillFn:
        if name in REGISTRY:
            raise ValueError(f"skill {name!r} already registered")
        REGISTRY[name] = fn
        return fn

    return register


def missing_skills(skill_names: set[str]) -> set[str]:
    """Skills a task graph uses that nobody registered; check before a run starts."""
    return skill_names - set(REGISTRY)
