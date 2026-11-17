"""Helpers shared by the expansion strategies.

The bounded retry transport moved to `mmregcheck.pillar2.llm.completion`; what remains here is
the response parsing that only the expansion strategies need, plus the truncated-completion
diagnostic shared by the sampling strategies.
"""

from __future__ import annotations

from typing import Any

from mmregcheck.pillar2.diagnostics import record_diagnostic


def sampled_texts(response: Any) -> list[tuple[int, str]]:
    """Return `(choice_index, stripped_text)` for every non-empty completion in `response`."""
    texts: list[tuple[int, str]] = []
    for index, choice in enumerate(response.choices):
        text = str(choice.message.content or "").strip()
        if text:
            texts.append((index, text))
    return texts


def record_truncated_completions(
    strategy_name: str,
    response: Any,
    *,
    max_tokens: int | None,
    diagnostics: list[str] | None,
) -> None:
    """Warn about and record every completion the endpoint cut off at its token budget.

    A `finish_reason` of `"length"` means the reasoning path is a prefix of what the model
    intended to write. The pipeline keeps the completion (the algorithm is unchanged), but the
    truncation must not stay invisible.

    Args:
        strategy_name: Name of the running strategy, used in the message.
        response: Chat-completion response whose choices carry `finish_reason`.
        max_tokens: Token budget that was requested, when the call set one.
        diagnostics: Run-level collector that receives the message.
    """
    for index, choice in enumerate(response.choices, start=1):
        if getattr(choice, "finish_reason", None) != "length":
            continue
        text = str(choice.message.content or "").strip()
        if not text:
            continue
        budget = f" at max_tokens={max_tokens}" if max_tokens is not None else ""
        record_diagnostic(
            diagnostics,
            f"{strategy_name}: completion {index} ended with finish_reason='length'{budget}; "
            "the reasoning path may be truncated.",
            stacklevel=3,
        )
