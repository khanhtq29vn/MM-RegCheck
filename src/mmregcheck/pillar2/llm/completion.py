"""Chat-completion transport helpers shared by the reasoning and explanation stages.

`create_completion` is the single bounded-retry call of the pipeline, so transport failures are
handled identically in expansion and explanation. `with_reasoning` moves the provider-specific
`reasoning` option into the request body for endpoints that accept it.
"""

from __future__ import annotations

import time
from typing import Any

from mmregcheck.pillar2.logger import TFBearLogger

# Retries must stay bounded, so the exponential backoff is capped instead of doubling forever.
_MAX_BACKOFF_SECONDS = 8.0


def with_reasoning(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Move `reasoning` into `extra_body` and merge any caller-supplied `extra_body`.

    The OpenAI SDK rejects unknown keyword arguments, and providers that support reasoning
    or custom chat-template flags expect them inside `extra_body`. A disabled `reasoning`
    option and an empty `extra_body` are omitted so strict OpenAI-compatible endpoints never
    see them.
    """
    out = dict(kwargs)
    reasoning = out.pop("reasoning", None)
    extra_body = out.pop("extra_body", None)
    merged_extra: dict[str, Any] = dict(extra_body) if extra_body else {}
    if reasoning and reasoning.get("enabled"):
        merged_extra["reasoning"] = reasoning
    if merged_extra:
        out["extra_body"] = merged_extra
    return out


def create_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_retry: int,
    logger: TFBearLogger,
    context: str,
    **kwargs: Any,
) -> Any:
    """Call the chat-completion endpoint, retrying at most `max_retry` times.

    The retry loop is bounded on purpose: a persistent failure must surface as one clear error
    instead of being hidden behind an outer retry (AGENTS 6.7). Only transient API errors are
    retried here; malformed model output is rejected by the parser of the calling strategy.
    """
    if max_retry < 1:
        raise ValueError(f"{context}: max_retry must be >= 1, got {max_retry}.")

    last_exc: Exception | None = None
    for attempt in range(1, max_retry + 1):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, **with_reasoning(kwargs)
            )
        except Exception as exc:
            last_exc = exc
            logger.log_warning(f"{context}: attempt {attempt}/{max_retry} failed: {exc}")
            if attempt < max_retry:
                time.sleep(min(2.0**attempt, _MAX_BACKOFF_SECONDS))

    raise RuntimeError(
        f"{context}: chat completion failed after {max_retry} attempts. Last error: {last_exc}"
    )
