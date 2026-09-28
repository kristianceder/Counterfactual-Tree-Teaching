"""The teacher: a hosted model (GPT-6 Luna) answering `FailureAnalyzer`'s prompts over the OpenAI
Responses API.

Each call sends the system prompt, the prompt's unchanging head behind an explicit cache breakpoint,
and the tail that changes (the experience text and the report), with reasoning off. A failed call is
retried by the SDK (twice, honouring retry-after), then up to `RETRIES` more times after a pause; a
request the API refuses outright (400, 401, 403, 404) is not. The robot keeps driving meanwhile:
the call runs on `AsyncFailureAnalyzer`'s worker thread, and a hosted model needs the interpreter
lock only to send the request and read the reply.

Every attempt is appended to a JSONL usage log (latency, token usage, cost), and the run stops once
it has spent `max_usd`. Needs `OPENAI_API_KEY` in the environment.
"""
from __future__ import annotations

import json
import os
import time

from .llm import FailureAnalyzer, LLMConfig

PRICES = {"gpt-6-luna": (0.10, 0.50)}
"""USD per million tokens (input, output), read 24 Sep 2026."""
CACHE_WRITE, CACHE_READ = 1.25, 0.10
"""Cache writes and reads, as multiples of the input price."""


def cost_usd(model: str, usage: dict) -> float:
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    return (usage["input_tokens"] * p_in + usage["cache_creation_input_tokens"] * p_in * CACHE_WRITE
            + usage["cache_read_input_tokens"] * p_in * CACHE_READ + usage["output_tokens"] * p_out) / 1e6


def call(client, model: str, system: str, head: str, tail: str, max_tokens: int,
         temperature: float | None = None) -> dict:
    """One call. Success: {"ok": True, "total_s", "usage", "usage_raw", "stop_reason", "text",
    "request_id"}; usage is normalised to input (full price) / cache_creation / cache_read / output
    tokens. Failure: {"ok": False, "total_s", "status" (when there was one), "error"}."""
    import openai
    t0 = time.perf_counter()
    try:
        # One breakpoint, after the head: the default implicit mode would add its own at the end of
        # the prompt and pay a cache write for the one-off report on every call.
        kwargs = dict(model=model, max_output_tokens=max_tokens, reasoning={"effort": "none"},
                      prompt_cache_options={"mode": "explicit"},
                      input=[{"role": "developer", "content": [{"type": "input_text", "text": system}]},
                             {"role": "user", "content": [
                                 {"type": "input_text", "text": head, "prompt_cache_breakpoint": {"mode": "explicit"}},
                                 {"type": "input_text", "text": tail}]}])
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = client.responses.create(**kwargs)
    except openai.APIStatusError as e:     # 4xx/5xx, among them 429 rate limit
        return dict(ok=False, total_s=time.perf_counter() - t0, status=e.status_code,
                    error=f"{e.status_code} {type(e).__name__}: {e.message}")
    except openai.APIError as e:           # connection failures, timeouts
        return dict(ok=False, total_s=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}")
    total = time.perf_counter() - t0
    u = response.usage
    read = u.input_tokens_details.cached_tokens or 0
    wrote = getattr(u.input_tokens_details, "cache_write_tokens", 0) or 0
    usage = dict(input_tokens=u.input_tokens - read - wrote, output_tokens=u.output_tokens,
                 cache_creation_input_tokens=wrote, cache_read_input_tokens=read)
    reason = getattr(response.incomplete_details, "reason", None)
    stop = ("end_turn" if response.status == "completed" else
            "max_tokens" if reason == "max_output_tokens" else f"{response.status}:{reason}")
    return dict(ok=True, ttft_s=None, total_s=total, usage=usage, usage_raw=u.model_dump(mode="json"),
                stop_reason=stop, text=response.output_text,
                request_id=getattr(response, "_request_id", None) or response.id)


class OpenAITeacher(FailureAnalyzer):
    """`FailureAnalyzer` whose `complete()` is a call to an OpenAI model."""
    RETRIES, PAUSE_S = 4, 20.0

    def __init__(self, config: LLMConfig, usage_log: str, max_usd: float = 2.0):
        super().__init__(config)
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set: export it (e.g. in ~/.bashrc, then open a new shell) and re-run")
        import openai
        self.client = openai.OpenAI(max_retries=2, timeout=30.0)
        self.usage_log = usage_log
        self.max_usd = max_usd
        self.spent_usd = 0.0

    def complete(self, system_prompt: str, head: str, tail: str) -> str:
        model = self.config.model_name
        for attempt in range(self.RETRIES + 1):
            result = call(self.client, model, system_prompt, head, tail,
                          max_tokens=self.config.max_output_tokens, temperature=self.config.temperature)
            cost = cost_usd(model, result["usage"]) if result["ok"] else 0.0
            self.spent_usd += cost
            with open(self.usage_log, "a") as log:
                log.write(json.dumps({"t": round(time.time(), 3), "attempt": attempt, "cost_usd": cost}
                                     | {k: v for k, v in result.items() if k != "text"}
                                     | ({"text": result["text"]} if result["ok"] else {})) + "\n")
            if result["ok"]:
                break
            if result.get("status") in (400, 401, 403, 404) or attempt == self.RETRIES:
                raise RuntimeError(f"teacher call failed after {attempt + 1} attempt(s): {result['error']}")
            time.sleep(self.PAUSE_S)
        u = result["usage"]
        self.last_prompt_tokens = u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"]
        self.last_cached_tokens = u["cache_read_input_tokens"]
        if self.spent_usd > self.max_usd:
            raise RuntimeError(f"teacher over budget: ${self.spent_usd:.2f} > ${self.max_usd:.2f}")
        return result["text"]

    def close(self) -> None:
        self.client.close()
