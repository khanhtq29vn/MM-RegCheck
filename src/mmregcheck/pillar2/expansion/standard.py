"""Standard (zero-shot) expansion strategy."""

from __future__ import annotations

from typing import Any

from mmregcheck.pillar2.base import BaseExpansionStrategy
from mmregcheck.pillar2.expansion._common import sampled_texts
from mmregcheck.pillar2.llm import create_completion
from mmregcheck.pillar2.pipeline_state import Node, TFBearState

#: Value of `Node.meta["strategy"]`.
STANDARD_STRATEGY = "standard"


class StandardExpansionStrategy(BaseExpansionStrategy):
    """Zero-shot baseline: one direct reasoning path per sampled completion.

    `temperature` defaults to 0.0 (greedy), so a request with `n_paths > 1` only produces
    diverse paths when the caller raises it; pass `temperature=0.2` to sample several
    independent paths from the same prompt.

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
            top_p=1.0,
            reasoning=self.reasoning,
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
            self.logger.log_info(f"Standard path {choice_index + 1}: {text}", style="dim")

        if not nodes:
            raise ValueError(f"{type(self).__name__}: the model returned no usable completion.")
        if len(nodes) < self.n_paths:
            raise ValueError(
                f"{type(self).__name__}: requested n_paths={self.n_paths} but the endpoint "
                f"returned only {len(nodes)} usable completion(s). The endpoint may ignore the "
                "`n` parameter; use one that supports it."
            )
        return nodes
