"""Tests for the per-domain prompt packs and their strategy/config wiring."""

from __future__ import annotations

import pytest

from mmregcheck.pillar2.expansion import (
    CoTSCExpansionStrategy,
    StandardExpansionStrategy,
    ToTExpansionStrategy,
)
from mmregcheck.pillar2.explainer import GenerativeExplanationStrategy
from mmregcheck.pillar2.pipeline_state import TFBearState, validate_template
from mmregcheck.pillar2.prior import LogprobPriorStrategy
from mmregcheck.pillar2.prompts import (
    EXPANSIONS,
    ExpansionName,
    load_prompt_pack,
    resolve_domain,
    resolve_expansion,
)
from mmregcheck.pillar2.scoring import CausalDeltaScoringStrategy
from mmregcheck.pillar2.tf_bear import TFBearAdjudicationResult
from mmregcheck.schemas import (
    ComplianceVerdict,
    RegulatoryDomain,
    TFBearBranchTrace,
)

DOMAINS = (RegulatoryDomain.BEAUTY, RegulatoryDomain.HEALTHCARE)
PACK_CASES = [(domain, expansion) for domain in DOMAINS for expansion in EXPANSIONS]
NON = ComplianceVerdict.NON_COMPLIANT


def _state() -> TFBearState:
    """A populated state so every placeholder has a non-empty value."""
    branch = TFBearBranchTrace(
        branch_id="P1",
        strategy="standard",
        prior_prob=0.2,
        posterior_prob=0.9,
        predicted_verdict=NON,
        rationale="TCA peels are outside a basic skincare scope.",
        posterior_distribution={NON: 0.9, ComplianceVerdict.PARTLY_COMPLIANT: 0.05},
    )
    return TFBearState(
        text_claim="A spa advertises a TCA chemical peel.",
        retrieved_regulatory_context="Basic Skincare Spa Scope: non-invasive cleansing only.",
        branches=[branch],
        adjudication=TFBearAdjudicationResult(
            final_verdict=NON,
            surviving_branches=[branch],
            pruned_branches=[],
            used_fallback=False,
        ),
    )


@pytest.mark.parametrize(("domain", "expansion"), PACK_CASES)
def test_every_pack_is_complete_and_validates(
    domain: RegulatoryDomain, expansion: ExpansionName
) -> None:
    pack = load_prompt_pack(domain, expansion)

    assert pack.domain is domain
    assert pack.expansion == expansion
    for name, template in (
        ("prior", pack.prior),
        ("expansion", pack.expansion_prompt),
        ("scoring", pack.scoring),
        ("explanation", pack.explanation),
    ):
        assert template.strip(), f"{domain.value}/{expansion}: empty {name} prompt"
        validate_template(template, f"PromptPack.{name}")

    if expansion == "tot":
        assert pack.tot_eval is not None
        validate_template(pack.tot_eval, "PromptPack.tot_eval")
    else:
        assert pack.tot_eval is None


@pytest.mark.parametrize(("domain", "expansion"), PACK_CASES)
def test_every_placeholder_renders_against_a_sample_state(
    domain: RegulatoryDomain, expansion: ExpansionName
) -> None:
    pack = load_prompt_pack(domain, expansion)
    state = _state()

    prior = state.render(pack.prior)
    assert "A spa advertises a TCA chemical peel." in prior

    if expansion == "tot":
        expansion_prompt = state.render(pack.expansion_prompt, state="first reasoning step")
        assert "first reasoning step" in expansion_prompt
        assert pack.tot_eval is not None
        evaluation = state.render(pack.tot_eval, state="first reasoning step")
        assert "first reasoning step" in evaluation
        assert '"evaluation"' in evaluation
        assert "{{" not in evaluation
    else:
        expansion_prompt = state.render(pack.expansion_prompt)
        assert "Basic Skincare Spa Scope" in expansion_prompt

    scoring = state.render(pack.scoring, state="the reasoning chain")
    assert "the reasoning chain" in scoring

    explanation = state.render(pack.explanation, paths="Path 1: reasoning")
    assert "Non-Compliant" in explanation
    assert "Path 1: reasoning" in explanation


@pytest.mark.parametrize("domain", DOMAINS)
def test_shared_prompts_are_defined_once_per_domain(domain: RegulatoryDomain) -> None:
    packs = [load_prompt_pack(domain, expansion) for expansion in EXPANSIONS]

    for field in ("prior", "scoring", "explanation"):
        values = {getattr(pack, field) for pack in packs}
        assert len(values) == 1, f"{field} differs across expansions of {domain.value}"


@pytest.mark.parametrize(("domain", "expansion"), PACK_CASES)
def test_legacy_placeholder_names_are_gone(
    domain: RegulatoryDomain, expansion: ExpansionName
) -> None:
    pack = load_prompt_pack(domain, expansion)
    templates = [pack.prior, pack.expansion_prompt, pack.scoring, pack.explanation]
    if pack.tot_eval is not None:
        templates.append(pack.tot_eval)

    for template in templates:
        assert "{claim}" not in template
        assert "{evidence}" not in template
        assert "{label}" not in template


def test_lookup_returns_the_requested_pack() -> None:
    assert load_prompt_pack(RegulatoryDomain.BEAUTY, "standard").domain is (RegulatoryDomain.BEAUTY)
    assert load_prompt_pack(RegulatoryDomain.HEALTHCARE, "tot").expansion == "tot"


def test_unknown_expansion_and_domain_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown expansion"):
        load_prompt_pack(RegulatoryDomain.BEAUTY, "cot")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown regulatory domain"):
        load_prompt_pack("Dental", "standard")  # type: ignore[arg-type]


def test_string_resolvers_accept_enum_values_and_names() -> None:
    assert resolve_domain("Beauty") is RegulatoryDomain.BEAUTY
    assert resolve_domain("healthcare") is RegulatoryDomain.HEALTHCARE
    assert resolve_domain("BEAUTY") is RegulatoryDomain.BEAUTY
    assert resolve_domain(RegulatoryDomain.HEALTHCARE) is RegulatoryDomain.HEALTHCARE
    with pytest.raises(ValueError, match="Unknown regulatory domain"):
        resolve_domain("Dental")

    assert resolve_expansion("standard") == "standard"
    assert resolve_expansion("cot-sc") == "cot-sc"
    assert resolve_expansion("tot") == "tot"
    with pytest.raises(ValueError, match="Unknown expansion"):
        resolve_expansion("cot")


@pytest.mark.parametrize(("domain", "expansion"), PACK_CASES)
def test_pack_builds_the_matching_strategy_types(
    domain: RegulatoryDomain, expansion: ExpansionName
) -> None:
    pack = load_prompt_pack(domain, expansion)
    prior, expansion_strategy, scoring, explanation = pack.build_strategies(
        explanation_max_tokens=128
    )

    assert isinstance(prior, LogprobPriorStrategy)
    assert prior.prompt_template == pack.prior
    assert isinstance(scoring, CausalDeltaScoringStrategy)
    assert scoring.prompt_template == pack.scoring
    assert isinstance(explanation, GenerativeExplanationStrategy)
    assert explanation.prompt_template == pack.explanation
    assert explanation.max_tokens == 128

    expected = {
        "standard": StandardExpansionStrategy,
        "cot-sc": CoTSCExpansionStrategy,
        "tot": ToTExpansionStrategy,
    }[expansion]
    assert isinstance(expansion_strategy, expected)
    assert expansion_strategy.prompt_template == pack.expansion_prompt
    if expansion == "tot":
        assert isinstance(expansion_strategy, ToTExpansionStrategy)
        assert expansion_strategy.eval_prompt_template == pack.tot_eval
        assert expansion_strategy.n_steps == 3
        assert expansion_strategy.k == 3
        assert expansion_strategy.n_evals == 2
        assert expansion_strategy.breadth_limit == 3
        assert expansion_strategy.min_steps == 2
        assert expansion_strategy.max_tokens_thought_gen == 1024
    else:
        assert expansion_strategy.n_paths == 3
        assert expansion_strategy.temperature == 0.7
        expected_max_tokens = 2000 if expansion == "cot-sc" else 512
        assert expansion_strategy.max_tokens == expected_max_tokens


def test_pack_expansion_kwargs_reflect_the_pack_fields() -> None:
    standard = load_prompt_pack(RegulatoryDomain.BEAUTY, "standard").expansion_kwargs()
    assert standard == {
        "prompt_template": load_prompt_pack(RegulatoryDomain.BEAUTY, "standard").expansion_prompt,
        "n_paths": 3,
        "max_tokens": 512,
        "temperature": 0.7,
    }

    tot = load_prompt_pack(RegulatoryDomain.HEALTHCARE, "tot").expansion_kwargs()
    assert tot["n_steps"] == 3
    assert tot["k"] == 3
    assert (
        tot["eval_prompt_template"] == load_prompt_pack(RegulatoryDomain.HEALTHCARE, "tot").tot_eval
    )


@pytest.mark.parametrize(("domain", "expansion"), PACK_CASES)
def test_to_config_wires_the_pack_and_keeps_the_thresholds(
    domain: RegulatoryDomain, expansion: ExpansionName
) -> None:
    pack = load_prompt_pack(domain, expansion)
    config = pack.to_config(tau_posterior=0.7, tau_delta=0.2, max_retry=5)

    assert config.prior_strategy is not None
    assert config.prior_strategy.prompt_template == pack.prior
    assert config.expansion_strategy is not None
    assert config.expansion_strategy.prompt_template == pack.expansion_prompt
    assert config.scoring_strategy is not None
    assert config.scoring_strategy.prompt_template == pack.scoring
    assert config.explanation_strategy is not None
    assert config.explanation_strategy.prompt_template == pack.explanation
    assert config.tau_posterior == 0.7
    assert config.tau_delta == 0.2
    assert config.max_retry == 5
    assert config.adjudication_strategy is not None
    assert config.adjudication_strategy.tau_posterior == 0.7


def test_label_mapping_is_preserved() -> None:
    for domain in DOMAINS:
        for expansion in EXPANSIONS:
            pack = load_prompt_pack(domain, expansion)
            combined = pack.prior + pack.scoring
            assert "output 0" in combined
            assert "output 1" in combined
            assert "output 2" in combined
            assert "Non-Compliant" in combined
            assert "Partly-Compliant" in combined
            assert "Compliant" in combined


def test_healthcare_glossary_is_copied_into_the_domain_prompts() -> None:
    marker = "Nâng mũi cấu trúc dựng trụ Nano Plus"
    last_line = "Thanh lý trái cây → Phẫu thuật loại bỏ ngực"

    for expansion in EXPANSIONS:
        pack = load_prompt_pack(RegulatoryDomain.HEALTHCARE, expansion)
        for template in (pack.expansion_prompt, pack.scoring):
            assert marker in template
            assert last_line in template

    beauty = load_prompt_pack(RegulatoryDomain.BEAUTY, "standard")
    assert marker not in beauty.expansion_prompt


def test_prompt_packs_contain_no_sample_data() -> None:
    forbidden = ("500lemon", "PT Beauty Spa", "ABC Aesthetic Clinic", "MEDREG-DEV-")
    for domain in DOMAINS:
        for expansion in EXPANSIONS:
            pack = load_prompt_pack(domain, expansion)
            templates = [pack.prior, pack.expansion_prompt, pack.scoring, pack.explanation]
            if pack.tot_eval is not None:
                templates.append(pack.tot_eval)
            for template in templates:
                for marker in forbidden:
                    assert marker not in template
