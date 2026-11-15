"""Abstract strategies of the TF-BEAR pipeline.

Every stage of the pipeline is an injectable strategy. To customise a stage, subclass the
matching `Base*Strategy` and pass an instance to the pipeline configuration; the subclass then
shares the logging, template validation and Delta computation of the built-in strategies.

    Stage 1  BasePriorStrategy         P_prior(y | E_T), before seeing the evidence
    Stage 2  BaseExpansionStrategy     reasoning paths conditioned on the evidence C-hat
    Stage 3  BaseScoringStrategy       P_posterior and Delta for each path
    Stage 4  BaseAdjudicationStrategy  dual-threshold gate and vote (`TFBearAdjudicator`)
    Stage 5  BaseExplanationStrategy   rationale for the final verdict

`client` is any chat-completion client the stage knows how to call (for example an OpenAI-
compatible SDK client); its concrete protocol lives next to the logprob elicitation code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mmregcheck.pillar2.logger import TFBearLogger
from mmregcheck.pillar2.pipeline_state import Node, TFBearState, validate_template
from mmregcheck.pillar2.results import TFBearAdjudicationResult
from mmregcheck.schemas import ComplianceVerdict, TFBearBranchTrace


class _BaseStrategy:
    """Shared scaffolding: prompt template validation, retry budget and logger."""

    def __init__(
        self,
        prompt_template: str = "",
        verbose: bool = False,
        max_retry: int = 3,
        reasoning: dict[str, Any] | None = None,
    ) -> None:
        self.prompt_template = prompt_template
        self.verbose = verbose
        self.max_retry = max_retry
        self.reasoning: dict[str, Any] = reasoning if reasoning is not None else {"enabled": False}
        self.logger = TFBearLogger(verbose=verbose)
        if prompt_template:
            validate_template(prompt_template, type(self).__name__)


class BasePriorStrategy(_BaseStrategy, ABC):
    """Computes P_prior; the template may reference `{text_claim}`."""

    @abstractmethod
    def compute_prior(
        self, state: TFBearState, client: Any, model: str
    ) -> dict[ComplianceVerdict, float]:
        """Return a verdict -> probability distribution using only `state.text_claim`."""


class BaseExpansionStrategy(_BaseStrategy, ABC):
    """Generates reasoning paths; the template may reference the claim, the context, `{state}`
    (tree search) and `{n_paths}` (self-consistency)."""

    @abstractmethod
    def expand(self, state: TFBearState, client: Any, model: str) -> list[Node]:
        """Return the reasoning paths for the claim and the regulatory context."""


class BaseScoringStrategy(_BaseStrategy, ABC):
    """Scores every path with P_posterior and Delta."""

    @abstractmethod
    def score(self, state: TFBearState, client: Any, model: str) -> list[TFBearBranchTrace]:
        """Return one trace per node of `state.nodes`.

        Delta must come from `TFBearBranchTrace.log_odds_delta`, the single epsilon-stabilised
        transform of the posterior and prior probabilities, so that every strategy reports a
        comparable Delta and scoring cannot disagree with adjudication.
        """


class BaseAdjudicationStrategy(_BaseStrategy, ABC):
    """Applies the dual-threshold gate and aggregates the surviving branches."""

    @abstractmethod
    def adjudicate(self, state: TFBearState, client: Any, model: str) -> TFBearAdjudicationResult:
        """Return the final verdict with the surviving and pruned branches."""


class BaseExplanationStrategy(_BaseStrategy, ABC):
    """Explains the final verdict; the template may reference `{verdict}` and `{paths}`."""

    @abstractmethod
    def explain(self, state: TFBearState, client: Any, model: str) -> str:
        """Return a natural-language explanation of `state.adjudication`."""
