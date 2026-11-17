"""Stage 5 explanation strategies: free-text and structured verdict rationales.

Both strategies render `prompt_template` through `TFBearState.render`, so a template may
reference `{text_claim}`, `{retrieved_regulatory_context}`, `{verdict}` and `{paths}`. The
`{paths}` block is built from the branches the adjudication stage kept, i.e. the reasoning the
verdict actually rests on:

    GenerativeExplanationStrategy            one free-text completion
    StructuredGenerativeExplanationStrategy  JSON-constrained completion validated by Pydantic

`synthesize_overall_explanation` aggregates technique-level verdicts with the strict hierarchy
for callers that do not run the LLM stages.

`client` is any OpenAI-compatible chat-completion client, i.e. an object exposing
`client.chat.completions.create(model=..., messages=[...], **kwargs)` whose response has
`.choices[0].message.content`.
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, Field

from mmregcheck.pillar2.base import BaseExplanationStrategy
from mmregcheck.pillar2.llm import create_completion, with_reasoning
from mmregcheck.pillar2.pipeline_state import TFBearState
from mmregcheck.schemas import TechniqueVerdictItem, TFBearBranchTrace, aggregate_strict_hierarchy

#: Default free-text prompt, written against the pipeline's placeholder names.
DEFAULT_EXPLANATION_PROMPT = """\
You are a regulatory compliance auditor.

Evidence: {retrieved_regulatory_context}
Claim: {text_claim}
Verdict: {verdict}

Supporting reasoning:
{paths}

Write 2-3 sentences explaining why the claim is {verdict}. Do not mention variable names."""

#: Default structured prompt; mirrors the free-text one but asks for the declared JSON fields.
DEFAULT_STRUCTURED_EXPLANATION_PROMPT = """\
You are a regulatory compliance auditor producing structured output.

Evidence: {retrieved_regulatory_context}
Claim: {text_claim}
Verdict: {verdict}

Supporting reasoning:
{paths}

Fill in the JSON output:
  verdict      : echo the verdict above exactly as written
  explanation  : 2-4 sentences explaining why the claim is {verdict}, citing specific evidence.
Do not mention variable names."""

#: Default name registered with the API for the structured-output schema.
DEFAULT_SCHEMA_NAME = "explanation_output"

# Retries must stay bounded, so the exponential backoff is capped instead of doubling forever.
_MAX_BACKOFF_SECONDS = 8.0


class ExplanationOutput(BaseModel):
    """Default output schema of `StructuredGenerativeExplanationStrategy`."""

    verdict: str = Field(description="Overall verdict; echo the pipeline verdict exactly.")
    explanation: str = Field(
        description="2-4 sentence regulatory rationale citing the retrieved evidence."
    )


def _supporting_paths(state: TFBearState) -> str:
    """Serialise the adjudicated branches as the `{paths}` template block.

    The explanation should cite the reasoning the verdict rests on, which is the set of
    surviving branches of the adjudication. Before adjudication has run (or when no branch
    passed the gate) every scored branch is listed instead, so the prompt is never silently
    empty.
    """
    adjudication = state.adjudication
    branches: list[TFBearBranchTrace]
    if adjudication is not None and adjudication.surviving_branches:
        branches = adjudication.surviving_branches
    else:
        branches = state.branches
    return "\n\n".join(
        f"Path {index}:\n{branch.rationale}" for index, branch in enumerate(branches, 1)
    )


def _parse_json_object(raw: str, *, context: str) -> dict[str, Any]:
    """Parse a structured completion, tolerating small syntax slips.

    Plain `json.loads` is tried first; `json_repair` is imported lazily only when that fails, so the
    core package does not require the optional `llm` extra. Output that cannot be parsed as an
    object raises `ValueError` and is retried by the calling strategy.
    """
    text = raw.strip()
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if not isinstance(parsed, dict):
        try:
            import json_repair
        except ImportError as exc:
            raise ValueError(
                f"{context}: could not parse model output as JSON ({raw[:200]!r}). "
                "Install the optional 'llm' extra for tolerant parsing via json-repair."
            ) from exc
        try:
            parsed = json_repair.loads(text)
        except Exception as exc:
            raise ValueError(
                f"{context}: could not parse model output as JSON ({raw[:200]!r})."
            ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{context}: model output is not a JSON object ({raw[:200]!r}).")
    return parsed


class GenerativeExplanationStrategy(BaseExplanationStrategy):
    """Explain the verdict with one free-text completion.

    Args:
        prompt_template: May reference `{text_claim}`, `{retrieved_regulatory_context}`,
            `{verdict}` and `{paths}`.
        max_tokens: Token budget of the explanation call; defaults to 1024.
        temperature: Sampling temperature; defaults to 0.3.
        top_p: Nucleus threshold; defaults to 0.95.
    """

    def __init__(
        self,
        prompt_template: str = DEFAULT_EXPLANATION_PROMPT,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.95,
        verbose: bool = False,
        max_retry: int = 3,
        reasoning: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            prompt_template=prompt_template,
            verbose=verbose,
            max_retry=max_retry,
            reasoning=reasoning,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def explain(self, state: TFBearState, client: Any, model: str) -> str:
        """Return the model's natural-language explanation of `state.adjudication`."""
        prompt = state.render(
            self.prompt_template,
            paths=_supporting_paths(state),
            verdict=state.verdict,
        )
        self.logger.log_prompt(prompt[:500], title="Prompt (first 500 chars)")

        response = create_completion(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_retry=self.max_retry,
            logger=self.logger,
            context=type(self).__name__,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            reasoning=self.reasoning,
            extra_body=self.extra_body,
        )
        if not response.choices:
            raise RuntimeError(f"{type(self).__name__}: the model returned no completion choice.")

        text = str(response.choices[0].message.content or "").strip()
        if not text:
            self.logger.log_warning(
                f"{type(self).__name__}: the model returned an empty explanation."
            )
        self.logger.log_explanation(text)
        return text


class StructuredGenerativeExplanationStrategy(BaseExplanationStrategy):
    """Explain the verdict with a JSON-constrained completion validated by Pydantic.

    The validated object is re-serialised to a JSON string, so callers can `json.loads()` the
    returned explanation to recover the structured fields.

    Args:
        output_schema: Pydantic model whose JSON schema is sent as the `response_format`.
        schema_name: Name registered with the API for the schema.
        prompt_template: May reference `{text_claim}`, `{retrieved_regulatory_context}`,
            `{verdict}` and `{paths}`.
        max_tokens: Token budget of the structured call; defaults to 1024.
        temperature: Sampling temperature; 0.0 keeps the JSON output deterministic.
        top_p: Nucleus threshold; defaults to 0.95.
    """

    def __init__(
        self,
        output_schema: type[BaseModel] = ExplanationOutput,
        schema_name: str = DEFAULT_SCHEMA_NAME,
        prompt_template: str = DEFAULT_STRUCTURED_EXPLANATION_PROMPT,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.95,
        verbose: bool = False,
        max_retry: int = 3,
        reasoning: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            prompt_template=prompt_template,
            verbose=verbose,
            max_retry=max_retry,
            reasoning=reasoning,
        )
        self.output_schema = output_schema
        self.schema_name = schema_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def explain(self, state: TFBearState, client: Any, model: str) -> str:
        """Return a JSON string conforming to `output_schema`.

        The retry budget covers the whole call -> parse -> validate cycle because a schema
        violation is an output failure, not a transport failure; the transport-only helper used by
        the other strategies would retry the wrong unit.
        """
        if self.max_retry < 1:
            raise ValueError(
                f"{type(self).__name__}: max_retry must be >= 1, got {self.max_retry}."
            )

        prompt = state.render(
            self.prompt_template,
            paths=_supporting_paths(state),
            verdict=state.verdict,
        )
        self.logger.log_prompt(prompt[:500], title="Prompt (first 500 chars)")

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": self.schema_name,
                "schema": self.output_schema.model_json_schema(),
            },
        }
        messages = [{"role": "user", "content": prompt}]

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retry + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **with_reasoning(
                        {
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                            "top_p": self.top_p,
                            "reasoning": self.reasoning,
                            "extra_body": self.extra_body,
                            "response_format": response_format,
                        }
                    ),
                )
                raw = str(response.choices[0].message.content or "").strip()
                parsed = _parse_json_object(raw, context=type(self).__name__)
                validated = self.output_schema.model_validate(parsed)
                result = validated.model_dump_json(indent=2)
            except Exception as exc:
                last_exc = exc
                self.logger.log_warning(
                    f"{type(self).__name__}: attempt {attempt}/{self.max_retry} failed: {exc}"
                )
                if attempt < self.max_retry:
                    time.sleep(min(2.0**attempt, _MAX_BACKOFF_SECONDS))
                continue

            self.logger.log_explanation(result)
            return result

        raise ValueError(
            f"{type(self).__name__}: failed after {self.max_retry} attempts. Last error: {last_exc}"
        )


def synthesize_overall_explanation(items: list[TechniqueVerdictItem]) -> tuple[str, str]:
    """Aggregate technique-level verdicts via strict hierarchy and format rationale."""
    overall = aggregate_strict_hierarchy([item.verdict for item in items])
    details = " ".join(
        f"[{item.technique_name}: {item.verdict.value}] {item.explanation}" for item in items
    )
    return overall.value, details.strip() or "No techniques extracted."
