"""Chain-of-thought self-consistency expansion strategy."""

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
COT_SC_STRATEGY = "cot-sc"


class CoTSCExpansionStrategy(BaseExpansionStrategy):
    """Generate `n_paths` independent chain-of-thought paths via API self-consistency sampling.

    The default sampling setting is `temperature=0.2`, `top_p=0.95` and `max_tokens=1024`. A
    temperature of 0.0 with more than one path is allowed but emits a `TFBearWarning`, because
    greedy sampling makes the paths likely to be identical.

    Args:
        n_paths: Number of independent completions requested from the API.
        max_tokens: Token budget per completion; defaults to 1024.
        temperature: Sampling temperature; defaults to 0.2.
        top_p: Nucleus threshold; defaults to 0.95.
        prompt_template: Chain-of-thought prompt.
        verbose: Enables verbose stage logging.
        max_retry: Retry budget of the LLM call.
    """

    def __init__(
        self,
        n_paths: int = 3,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.95,
        prompt_template: str = "",
        verbose: bool = False,
        max_retry: int = 3,
    ) -> None:
        super().__init__(prompt_template=prompt_template, verbose=verbose, max_retry=max_retry)
        self.n_paths = n_paths
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def expand(self, state: TFBearState, client: Any, model: str) -> list[Node]:
        """Return one terminal `Node` per sampled chain-of-thought path."""
        if self.n_paths > 1 and self.temperature == 0.0:
            warnings.warn(
                f"{type(self).__name__}: n_paths={self.n_paths} with temperature=0.0; "
                "the sampled paths may be identical.",
                TFBearWarning,
                stacklevel=2,
            )

        prompt = state.render(self.prompt_template, n_paths=self.n_paths)
        self.logger.log_info(f"Prompt:\n{prompt}", style="dim")
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
                    meta={"strategy": COT_SC_STRATEGY, "cot_index": choice_index},
                )
            )
            self.logger.log_info(f"CoT-SC path {choice_index + 1}: {text}", style="dim")

        if not nodes:
            raise ValueError(f"{type(self).__name__}: the model returned no usable completion.")
        if len(nodes) != self.n_paths:
            raise ValueError(
                f"{type(self).__name__}: requested n_paths={self.n_paths} but the endpoint "
                f"returned {len(nodes)} usable completion(s). The endpoint must return exactly "
                "n_paths usable completions; use one that honours the `n` parameter."
            )
        return nodes
