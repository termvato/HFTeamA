"""Step check: after an arm reports a step, a fresh camera frame and the same VLM confirm it.

The arm's own "ok" is not fully trusted (architecture doc, section 7). Fail safe: an answer that
cannot be parsed after the retries counts as not confirmed, never as done.
The executor decides what happens next (retry, re-split); this module only answers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from hfteama.perception.decomposer import Bin
from hfteama.perception.prompts import SYSTEM, check_prompt
from hfteama.perception.vlm import VLM, ParseError, VLMError, parse_json
from hfteama.taskgraph import Task


@dataclass(frozen=True)
class CheckResult:
    done: bool
    confidence: float
    reason: str = ""


def question_for(task: Task, bins: dict[str, Bin]) -> str:
    a = task.args
    what = a.get("object", "the object")
    if a.get("category"):
        what = f"the {a['category']} object {what}"
    if task.skill == "place" and a.get("target"):
        target = bins[a["target"]].label if a["target"] in bins else a["target"]
        return f"Is {what} now inside {target}?"
    if task.skill == "pick":
        return f"Is {what} held in a robot gripper, lifted off the table?"
    return f"Did the step '{task.skill}' on {what} succeed?"


class StepChecker:
    def __init__(
        self, vlm: VLM, bins: list[Bin], threshold: float = 0.6, max_retries: int = 1
    ) -> None:
        self.vlm = vlm
        self.bins = {b.id: b for b in bins}
        self.threshold = threshold
        self.max_retries = max_retries

    def check(self, task: Task, image: Any) -> CheckResult:
        question = question_for(task, self.bins)
        errors: list[str] = []
        for _ in range(self.max_retries + 1):
            try:
                text = self.vlm.ask(image, check_prompt(question, errors or None), SYSTEM)
            except VLMError as e:
                return CheckResult(False, 0.0, f"check_error: {e}")
            try:
                d = parse_json(text)
                if not isinstance(d.get("done"), bool):
                    raise ParseError('"done" must be true or false')
                conf = float(d.get("confidence", 1.0 if d["done"] else 0.0))
            except (ParseError, TypeError, ValueError) as e:
                errors = [str(e)]
                continue
            conf = min(max(conf, 0.0), 1.0)
            done = d["done"] and conf >= self.threshold
            reason = str(d.get("reason", ""))
            if d["done"] and not done:
                reason = f"not sure ({conf:.2f}): {reason}".rstrip(": ")
            return CheckResult(done, conf, reason)
        return CheckResult(False, 0.0, "check_unparsed: " + "; ".join(errors))

    def as_verifier(self, grab: Callable[[], Any]) -> Callable[[Task, Any], tuple[bool, str]]:
        """The executor's step-check hook: Executor(verifier=checker.as_verifier(camera.grab)).

        Grabs a fresh frame after each successful step and asks the VLM. A camera error counts
        as not confirmed.
        """

        def verify(task: Task, result: Any) -> tuple[bool, str]:
            try:
                image = grab()
            except Exception as e:  # camera unplugged, driver error, ...
                return False, f"camera error: {e}"
            r = self.check(task, image)
            return r.done, r.reason or f"confidence {r.confidence:.2f}"

        return verify
