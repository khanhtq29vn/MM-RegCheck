"""Public entry point of the TF-BEAR pipeline.

The pipeline runs five stages over one claim and its retrieved regulatory context:

    Stage 1  Prior        `mmregcheck.pillar2.prior`        P_prior(y | E_T)
    Stage 2  Expansion    `mmregcheck.pillar2.expansion`    reasoning paths conditioned on C-hat
    Stage 3  Scoring      `mmregcheck.pillar2.scoring`      P_posterior and Delta per path
    Stage 4  Adjudication `mmregcheck.pillar2.adjudication` dual-threshold gate and vote
    Stage 5  Explanation  `mmregcheck.pillar2.explainer`    rationale for the final verdict

This module is orchestration only: it wires a `TFBearConfig` and an optional set of per-stage
clients into the engine. The orchestration itself lives in `mmregcheck.pillar2.engine`, the
strategy contracts in `mmregcheck.pillar2.base`, the shared state in
`mmregcheck.pillar2.pipeline_state`, and the result types in `mmregcheck.pillar2.results`.

    bear = TFBear.from_domain("Beauty", "standard", endpoint=EndpointConfig(...))
    result = bear.run("A spa advertises a TCA chemical peel.", "Basic skincare scope only.")
"""

from __future__ import annotations

from typing import Any

from mmregcheck.pillar2.adjudication import TFBearAdjudicator
from mmregcheck.pillar2.config import TFBearConfig
from mmregcheck.pillar2.engine import StageClients, run_tfbear
from mmregcheck.pillar2.llm import EndpointConfig
from mmregcheck.pillar2.pipeline_state import TFBearState
from mmregcheck.pillar2.prompts import (
    ExpansionName,
    load_prompt_pack,
    resolve_domain,
    resolve_expansion,
)
from mmregcheck.pillar2.results import TFBearAdjudicationResult, TFBearResult
from mmregcheck.pillar2.sample_adapter import state_from_sample, to_submission_prediction
from mmregcheck.schemas import MedRegSampleInput, RegulatoryDomain, SubmissionPrediction

__all__ = [
    "TFBear",
    "TFBearAdjudicationResult",
    "TFBearAdjudicator",
    "TFBearConfig",
    "TFBearResult",
    "run_tfbear",
]


class TFBear:
    """The TF-BEAR pipeline: configure once, then run claims or benchmark samples.

    Args:
        config: Strategy wiring, endpoint selection, thresholds and global settings.
        clients: Optional per-stage clients. When omitted, one client is built per stage from
            the configured endpoints on the first run, which requires the endpoint's API-key
            variable to be set.
    """

    def __init__(self, config: TFBearConfig, *, clients: StageClients | None = None) -> None:
        self.config = config
        self._clients = clients

    @classmethod
    def from_domain(
        cls,
        domain: RegulatoryDomain | str,
        expansion: ExpansionName = "standard",
        *,
        endpoint: EndpointConfig | None = None,
        **config_overrides: Any,
    ) -> TFBear:
        """Build a pipeline from one of the per-domain prompt packs.

        Args:
            domain: Benchmark domain, as an enum value or its case-insensitive name.
            expansion: One of `"standard"`, `"cot-sc"` or `"tot"`.
            endpoint: Default endpoint of every stage.
            **config_overrides: Forwarded to `PromptPack.to_config` (thresholds, verbosity,
                retry budget, reasoning payload, per-stage endpoints, explanation budget).

        Raises:
            ValueError: If the domain or the expansion variant is unknown.
        """
        pack = load_prompt_pack(resolve_domain(domain), resolve_expansion(expansion))
        return cls(pack.to_config(endpoint=endpoint, **config_overrides))

    def run(self, text_claim: str, retrieved_regulatory_context: str = "") -> TFBearResult:
        """Run the five stages over one claim and its retrieved regulatory context."""
        return self._run_state(
            TFBearState(
                text_claim=text_claim,
                retrieved_regulatory_context=retrieved_regulatory_context,
            )
        )

    def run_sample(
        self,
        sample: MedRegSampleInput,
        *,
        open_book: bool = True,
    ) -> SubmissionPrediction:
        """Run one benchmark sample and return a leaderboard prediction.

        With `open_book=False` the sample's retrieved regulatory context is dropped before the
        run, which gives the close-book (evidence-free) condition: the pipeline must decide from
        the claim alone.
        """
        state = state_from_sample(sample)
        if not open_book:
            state.retrieved_regulatory_context = ""
        result = self._run_state(state)
        return to_submission_prediction(result, sample.sample_id)

    def _run_state(self, state: TFBearState) -> TFBearResult:
        return run_tfbear(state, self.config, clients=self._clients)
