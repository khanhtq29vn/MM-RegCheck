"""Chain-of-thought self-consistency expansion strategy."""

from __future__ import annotations

from typing import Any

from mmregcheck.pillar2.base import BaseExpansionStrategy
from mmregcheck.pillar2.expansion._common import _create_completion, _sampled_texts
from mmregcheck.pillar2.pipeline_state import Node, TFBearState

#: Value of `Node.meta["strategy"]`.
COT_SC_STRATEGY = "cot-sc"


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
