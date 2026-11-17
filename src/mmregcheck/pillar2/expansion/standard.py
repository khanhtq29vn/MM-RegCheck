"""Standard (zero-shot) expansion strategy."""

from __future__ import annotations

import warnings
from typing import Any

from mmregcheck.pillar2.base import BaseExpansionStrategy
from mmregcheck.pillar2.diagnostics import TFBearWarning
from mmregcheck.pillar2.expansion._common import (
    record_truncated_completions,
    sampled_texts,
)
from mmregcheck.pillar2.llm import create_completion
from mmregcheck.pillar2.pipeline_state import Node, TFBearState

#: Value of `Node.meta["strategy"]`.
STANDARD_STRATEGY = "standard"


class StandardExpansionStrategy(BaseExpansionStrategy):
    """Zero-shot baseline: one direct reasoning path per sampled completion.

    The default sampling setting is `temperature=0.2`, `top_p=0.95` and `max_tokens=1024`, so a
    request with `n_paths > 1` produces diverse paths. A temperature of 0.0 with more than one
    path is allowed but emits a `TFBearWarning`, because greedy sampling makes the paths likely
    to be identical.

    Args:
        n_paths: Number of independent completions requested from the API.
        max_tokens: Token budget per completion; defaults to 1024.
        temperature: Sampling temperature; defaults to 0.2.
        top_p: Nucleus threshold; defaults to 0.95.
        prompt_template: Direct reasoning prompt.
        verbose: Enables verbose stage logging.
        max_retry: Retry budget of the LLM call.
    """

    def __init__(
        self,
        n_paths: int = 1,
        prompt_template: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.95,
        verbose: bool = False,
        max_retry: int = 3,
    ) -> None:
        super().__init__(prompt_template=prompt_template, verbose=verbose, max_retry=max_retry)
        self.n_paths = n_paths
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def expand(self, state: TFBearState, client: Any, model: str) -> list[Node]:
        """Return one terminal `Node` per non-empty completion of the direct prompt."""
        if self.n_paths > 1 and self.temperature == 0.0:
            warnings.warn(
                f"{type(self).__name__}: n_paths={self.n_paths} with temperature=0.0; "
                "the sampled paths may be identical.",
                TFBearWarning,
                stacklevel=2,
            )

        prompt = state.render(self.prompt_template)
        self.logger.log_prompt(prompt)
        messages = [{"role": "user", "content": prompt}]

        response = create_completion(
            client,
            model=model,
            messages=messages,
            max_retry=self.max_retry,
            logger=self.logger,
            context=type(self).__name__,
            n=self.n_paths,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            reasoning=self.reasoning,
            extra_body=self.extra_body,
        )
        record_truncated_completions(
            type(self).__name__,
            response,
            max_tokens=self.max_tokens,
            diagnostics=state.diagnostics,
        )

        nodes: list[Node] = []
        for choice_index, text in sampled_texts(response):
            nodes.append(
                Node(
                    state=text,
                    thought=text,
                    is_terminal=True,
                    meta={"strategy": STANDARD_STRATEGY, "choice_index": choice_index},
                )
            )
            self.logger.log_response(text, title=f"Response (Standard path {choice_index + 1})")

        if not nodes:
            raise ValueError(f"{type(self).__name__}: the model returned no usable completion.")
        if len(nodes) != self.n_paths:
            raise ValueError(
                f"{type(self).__name__}: requested n_paths={self.n_paths} but the endpoint "
                f"returned {len(nodes)} usable completion(s). The endpoint must return exactly "
                "n_paths usable completions; use one that honours the `n` parameter."
            )
        return nodes
