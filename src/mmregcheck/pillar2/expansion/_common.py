"""Helpers shared by the expansion strategies.

The bounded retry transport moved to `mmregcheck.pillar2.llm.completion`; what remains here is
the response parsing that only the expansion strategies need.
"""

from __future__ import annotations

from typing import Any


def sampled_texts(response: Any) -> list[tuple[int, str]]:
    """Return `(choice_index, stripped_text)` for every non-empty completion in `response`."""
    texts: list[tuple[int, str]] = []
    for index, choice in enumerate(response.choices):
        text = str(choice.message.content or "").strip()
        if text:
            texts.append((index, text))
    return texts
