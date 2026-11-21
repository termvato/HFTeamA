"""VLM client: one call, image + prompt -> text. Model-agnostic; swapping models is one config line.

- QwenVLM: Qwen vision-language model from Hugging Face, run locally with transformers
  (pip install -e ".[vlm]"). Zero-shot, no fine-tuning. Loaded on first use, with safeguards
  for a laptop GPU (sized for an RTX 4060 Laptop, 8 GB).
- FakeVLM: scripted answers, for tests and for running the loop without a GPU.

Every failure of the model itself (no GPU, not enough memory, too hot, out of memory) raises
VLMError; callers turn it into "not confirmed" (step check) or "ask the operator" (decomposer).
"""

from __future__ import annotations

import gc
import json
import re
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any, Protocol

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


class VLMError(RuntimeError):
    """The model could not answer (GPU missing, memory, temperature, load error)."""


class VLM(Protocol):
    def ask(self, image: Any, prompt: str, system: str | None = None) -> str:
        """Answer `prompt` about `image` (PIL image). Returns the raw model text."""
        ...


def gpu_temperature_c() -> float | None:
    """GPU 0 temperature from nvidia-smi, or None if it cannot be read."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        return float(out.stdout.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def memory_plan(
    free_gb: float, total_gb: float, params_b: float, quantize: str, vram_fraction: float
) -> str:
    """Pick "bf16" or "4bit" for the VRAM budget; raise VLMError if neither fits."""
    budget = min(free_gb, total_gb * vram_fraction)
    headroom = 1.5  # activations, vision tokens, KV cache for a short answer
    need = {"bf16": 2.0 * params_b + headroom, "4bit": 0.75 * params_b + headroom}
    options = {"none": ["bf16"], "4bit": ["4bit"], "auto": ["bf16", "4bit"]}[quantize]
    for mode in options:
        if need[mode] <= budget:
            return mode
    raise VLMError(
        f"not enough GPU memory: {budget:.1f} GB usable ({free_gb:.1f} GB free of "
        f"{total_gb:.1f} GB), needs {need[options[-1]]:.1f} GB as {options[-1]}. "
        "Close other GPU programs or use a smaller model."
    )


class QwenVLM:
    """Qwen VL via Hugging Face transformers, with GPU safeguards. Greedy decoding (repeatable).

    Safeguards:
    - Load: needs CUDA (CPU only if asked for with device="cpu"); checks free VRAM against an
      estimate before loading; caps this process at `vram_fraction` of the card; puts the whole
      model on the GPU (no silent CPU offload, which would make every answer crawl).
    - quantize="auto": bf16 if it fits the budget, else 4-bit NF4 (vision tower kept in bf16).
    - Each call: one at a time (lock); waits for the GPU to cool below `max_temp_c` (up to
      `cool_wait_s`), else refuses; image downscaled to `max_image_side`; output capped at
      `max_new_tokens` and `max_time_s`; on CUDA out-of-memory, frees memory and retries once
      at half the image size, then raises VLMError.
    - unload() frees the GPU.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        quantize: str = "auto",  # "auto" | "4bit" | "none"
        device: str = "cuda",
        params_b: float = 4.4,  # model size in billions, for the memory estimate
        vram_fraction: float = 0.85,
        max_new_tokens: int = 512,
        max_time_s: float = 45.0,
        max_image_side: int = 1024,
        max_temp_c: float = 85.0,
        cool_wait_s: float = 30.0,
    ) -> None:
        if quantize not in ("auto", "4bit", "none"):
            raise ValueError("quantize must be 'auto', '4bit' or 'none'")
        self.model_id = model_id
        self.quantize = quantize
        self.device = device
        self.params_b = params_b
        self.vram_fraction = vram_fraction
        self.max_new_tokens = max_new_tokens
        self.max_time_s = max_time_s
        self.max_image_side = max_image_side
        self.max_temp_c = max_temp_c
        self.cool_wait_s = cool_wait_s
        self.loaded_as: str | None = None  # "bf16" | "4bit" | "cpu" once loaded
        self.last_stats: dict[str, float] = {}
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.Lock()

    # --- loading -------------------------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:  # pragma: no cover - depends on the machine
            raise VLMError('QwenVLM needs torch and transformers: pip install -e ".[vlm]"') from e

        kwargs: dict[str, Any] = {}
        if self.device == "cpu":
            mode = "cpu"
            kwargs.update(dtype=torch.float32, device_map={"": "cpu"})
        else:
            if not torch.cuda.is_available():
                raise VLMError("no CUDA GPU found (pass device='cpu' to run slowly on the CPU)")
            free, total = (v / 2**30 for v in torch.cuda.mem_get_info(0))
            mode = memory_plan(free, total, self.params_b, self.quantize, self.vram_fraction)
            torch.cuda.set_per_process_memory_fraction(self.vram_fraction, 0)
            kwargs.update(dtype=torch.bfloat16, device_map={"": 0})
            if mode == "4bit":
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=["visual", "lm_head"],
                )
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)
            self._model.eval()
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception as e:
            self.unload()
            raise VLMError(f"could not load {self.model_id} ({mode}): {e}") from e
        self.loaded_as = mode

    def unload(self) -> None:
        """Free the model and its GPU memory."""
        self._model = None
        self._processor = None
        self.loaded_as = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    # --- answering -----------------------------------------------------------------------
    def _wait_cool(self) -> None:
        if self.device == "cpu":
            return
        deadline = time.monotonic() + self.cool_wait_s
        while True:
            t = gpu_temperature_c()
            if t is None or t < self.max_temp_c:
                return
            if time.monotonic() >= deadline:
                raise VLMError(f"GPU at {t:.0f} C, above the {self.max_temp_c:.0f} C limit")
            time.sleep(2.0)

    @staticmethod
    def _shrink(image: Any, side: int) -> Any:
        from hfteama.perception.annotate import to_pil

        img = to_pil(image)
        if max(img.size) > side:
            img = img.copy()
            img.thumbnail((side, side))
        return img

    def _generate(self, image: Any, prompt: str, system: str | None) -> str:
        import torch

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            }
        )
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                max_time=self.max_time_s,
                do_sample=False,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1] :]
        self.last_stats["new_tokens"] = float(new_tokens.shape[1])
        return self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def ask(self, image: Any, prompt: str, system: str | None = None) -> str:
        if not self._lock.acquire(timeout=self.max_time_s * 2 + self.cool_wait_s):
            raise VLMError("VLM busy: another call did not finish in time")
        try:
            self._load()
            self._wait_cool()
            import torch

            side = self.max_image_side
            for attempt in (1, 2):
                t0 = time.monotonic()
                try:
                    text = self._generate(self._shrink(image, side), prompt, system)
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    if attempt == 2:
                        raise VLMError("GPU out of memory, even at half image size") from e
                    side //= 2
                    continue
                self.last_stats.update(seconds=time.monotonic() - t0, image_side=float(side))
                if torch.cuda.is_available():
                    self.last_stats["peak_vram_gb"] = torch.cuda.max_memory_allocated(0) / 2**30
                return text
            raise VLMError("no answer")  # pragma: no cover
        finally:
            self._lock.release()


class FakeVLM:
    """Returns scripted answers in order, or from a function of the prompt. Records every call.

    An answer that is an exception instance is raised instead (to test VLMError handling).
    """

    def __init__(self, answers: Iterable[str | Exception] | Callable[[str], str]) -> None:
        self._fn = answers if callable(answers) else None
        self._queue = [] if callable(answers) else list(answers)
        self.calls: list[str] = []

    def ask(self, image: Any, prompt: str, system: str | None = None) -> str:
        self.calls.append(prompt)
        if self._fn is not None:
            return self._fn(prompt)
        if not self._queue:
            raise RuntimeError("FakeVLM ran out of scripted answers")
        a = self._queue.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


class ParseError(ValueError):
    pass


def parse_json(text: str) -> dict[str, Any]:
    """The JSON object in a model answer (tolerates <think> blocks and ```json fences)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ParseError("no JSON object in the answer")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e.msg}") from e
    if not isinstance(value, dict):
        raise ParseError("answer is not a JSON object")
    return value
