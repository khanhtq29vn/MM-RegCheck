"""Reasoning-path expansion strategies of the TF-BEAR pipeline (Section 4.2).

Port of the TF-BEAR draft `strategies/expansion.py` reduced to the two non-tree strategies:

    Standard     one direct reasoning path per sampled completion (zero-shot baseline)
    CoT-SC       `n_paths` sampled chain-of-thought paths, aggregated by self-consistency

The tree-search strategy lives in `mmregcheck.pillar2.tot`. Every strategy returns
`mmregcheck.pillar2.pipeline_state.Node` objects and stores the strategy name in
`Node.meta["strategy"]`, which downstream stages copy into `TFBearBranchTrace.strategy`.

`client` is any OpenAI-compatible chat-completion client, i.e. an object exposing
`client.chat.completions.create(model=..., messages=[...], **kwargs)` whose response has
`.choices[i].message.content`.
"""

from __future__ import annotations

import time
from typing import Any

from mmregcheck.pillar2.base import BaseExpansionStrategy
from mmregcheck.pillar2.logger import TFBearLogger
from mmregcheck.pillar2.pipeline_state import Node, TFBearState

#: Values of `Node.meta["strategy"]`; the scoring stage copies them into
#: `TFBearBranchTrace.strategy`.
STANDARD_STRATEGY = "standard"
COT_SC_STRATEGY = "cot-sc"

# Retries must stay bounded, so the exponential backoff is capped instead of doubling forever.
_MAX_BACKOFF_SECONDS = 8.0


def _create_completion(
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
            return client.chat.completions.create(model=model, messages=messages, **kwargs)
        except Exception as exc:
            last_exc = exc
            logger.log_warning(f"{context}: attempt {attempt}/{max_retry} failed: {exc}")
            if attempt < max_retry:
                time.sleep(min(2.0**attempt, _MAX_BACKOFF_SECONDS))

    raise RuntimeError(
        f"{context}: chat completion failed after {max_retry} attempts. Last error: {last_exc}"
    )


def _sampled_texts(response: Any) -> list[tuple[int, str]]:
    """Return `(choice_index, stripped_text)` for every non-empty completion in `response`."""
    texts: list[tuple[int, str]] = []
    for index, choice in enumerate(response.choices):
        text = str(choice.message.content or "").strip()
        if text:
            texts.append((index, text))
    return texts


class StandardExpansionStrategy(BaseExpansionStrategy):
    """Zero-shot baseline: one direct reasoning path per sampled completion.

    Parameters mirror the TF-BEAR draft. `temperature` defaults to 0.0 (greedy), so a request with
    `n_paths > 1` only produces diverse paths when the caller raises it. The paper's Appendix
    reports `T = 0.2` for reasoning expansion; pass `temperature=0.2` to reproduce that setting.

    Args:
        n_paths: Number of independent completions requested from the API.
        max_tokens: Token budget per completion.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        n_paths: int = 1,
        prompt_template: str = "",
        max_tokens: int = 512,
        temperature: float = 0.0,
        verbose: bool = False,
        max_retry: int = 3,
    ) -> None:
        super().__init__(prompt_template=prompt_template, verbose=verbose, max_retry=max_retry)
        self.n_paths = n_paths
        self.max_tokens = max_tokens
        self.temperature = temperature

    def expand(self, state: TFBearState, client: Any, model: str) -> list[Node]:
        """Return one terminal `Node` per non-empty completion of the direct prompt."""
        prompt = state.render(self.prompt_template)
        self.logger.log_info(f"Prompt:\n{prompt}", style="dim")
        messages = [{"role": "user", "content": prompt}]

        response = _create_completion(
            client,
            model=model,
            messages=messages,
            max_retry=self.max_retry,
            logger=self.logger,
            context=type(self).__name__,
            n=self.n_paths,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=1.0,
            reasoning=self.reasoning,
        )

        nodes: list[Node] = []
        for choice_index, text in _sampled_texts(response):
            nodes.append(
                Node(
                    state=text,
                    thought=text,
                    is_terminal=True,
                    meta={"strategy": STANDARD_STRATEGY, "choice_index": choice_index},
                )
            )
            self.logger.log_info(f"Standard path {choice_index + 1}: {text}", style="dim")

        if not nodes:
            raise ValueError(f"{type(self).__name__}: the model returned no usable completion.")
        return nodes


class CoTSCExpansionStrategy(BaseExpansionStrategy):
    """Generate `n_paths` independent chain-of-thought paths via API self-consistency sampling.

    Parameters mirror the TF-BEAR draft. `temperature` defaults to 0.7; the paper's Appendix
    reports `T = 0.2` for reasoning expansion, which a caller can request explicitly.

    Args:
        n_paths: Number of independent completions requested from the API.
        max_tokens: Token budget per completion.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        n_paths: int = 3,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        prompt_template: str = "",
        verbose: bool = False,
        max_retry: int = 3,
    ) -> None:
        super().__init__(prompt_template=prompt_template, verbose=verbose, max_retry=max_retry)
        self.n_paths = n_paths
        self.max_tokens = max_tokens
        self.temperature = temperature

    def expand(self, state: TFBearState, client: Any, model: str) -> list[Node]:
        """Return one terminal `Node` per sampled chain-of-thought path."""
        prompt = state.render(self.prompt_template, n_paths=self.n_paths)
        self.logger.log_info(f"Prompt:\n{prompt}", style="dim")
        messages = [{"role": "user", "content": prompt}]

        response = _create_completion(
            client,
            model=model,
            messages=messages,
            max_retry=self.max_retry,
            logger=self.logger,
            context=type(self).__name__,
            n=self.n_paths,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=1.0,
            reasoning=self.reasoning,
        )

        nodes: list[Node] = []
        for choice_index, text in _sampled_texts(response):
            nodes.append(
                Node(
                    state=text,
                    thought=text,
                    is_terminal=True,
                    meta={"strategy": COT_SC_STRATEGY, "cot_index": choice_index},
                )
            )
            self.logger.log_info(f"CoT-SC path {choice_index + 1}: {text}", style="dim")

        if not nodes:
            raise ValueError(f"{type(self).__name__}: the model returned no usable completion.")
        return nodes
