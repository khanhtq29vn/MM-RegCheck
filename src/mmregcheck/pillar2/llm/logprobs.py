"""Elicitation of the three-way compliance distribution from label-token logprobs.

The model answer is constrained to a single digit `y_t in {0, 1, 2}`. One chat call with
`max_tokens=LABEL_TOKEN_MAX_TOKENS` and logprobs therefore yields a distribution over the three
`ComplianceVerdict` members directly, without parsing free text.

This module is the single implementation the prior and scoring stages share, so the prior and
the posterior are elicited with exactly the same procedure and remain comparable. Sampling is
caller-controlled: `temperature`, `top_p` and `top_logprobs` are arguments whose defaults are the
documented elicitation settings.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from mmregcheck.pillar2.diagnostics import record_diagnostic
from mmregcheck.pillar2.llm.clients import ChatClient, ChoiceLogprobs
from mmregcheck.schemas import ComplianceVerdict

#: The label-token protocol emits exactly one token per call. The value is part of the protocol,
#: not a sampling budget, so it is deliberately not caller-configurable.
LABEL_TOKEN_MAX_TOKENS = 1

# Benchmark label mapping: the digit the model emits maps onto the schema's verdict values in
# this order. Generic label sets such as "Refuted"/"Not Enough Info"/"Supported" do not name
# `ComplianceVerdict` members and are deliberately not used.
TOKEN_TO_VERDICT: dict[str, ComplianceVerdict] = {
    "0": ComplianceVerdict.NON_COMPLIANT,
    "1": ComplianceVerdict.PARTLY_COMPLIANT,
    "2": ComplianceVerdict.COMPLIANT,
}


class LogprobElicitationError(RuntimeError):
    """Raised when a label distribution cannot be elicited from the endpoint."""


class MissingLogprobsError(LogprobElicitationError):
    """Raised when the response carries no log-probability payload at all."""


class LabelTokenMissingError(LogprobElicitationError):
    """Raised when none of the label tokens appears in the returned top logprobs.

    When no label is present there is nothing to renormalise, so a distribution would have to be
    invented; this error prevents a fabricated Delta downstream.
    """


def _top_logprobs_by_token(logprobs: ChoiceLogprobs | None) -> dict[str, float]:
    """Flatten the first generated position's candidates into a token -> logprob mapping."""
    if logprobs is None:
        raise MissingLogprobsError(
            "The chat completion returned no logprobs; request it with logprobs=True."
        )
    content = logprobs.content
    if not content:
        raise MissingLogprobsError("The chat completion returned no logprob content.")
    candidates = content[0].top_logprobs
    if not candidates:
        raise MissingLogprobsError("The chat completion returned an empty top_logprobs list.")
    return {entry.token: float(entry.logprob) for entry in candidates}


def _restricted_softmax(
    top_logprobs: Mapping[str, float],
    label_tokens: Mapping[str, ComplianceVerdict],
) -> dict[ComplianceVerdict, float]:
    """Softmax over the label tokens that the endpoint returned.

    A label outside the top-K candidates gets probability 0.0 and the present labels are
    renormalised, so the result is always a proper softmax over what the model actually emitted.
    Tokens may carry a leading space.
    """
    log_probs: dict[ComplianceVerdict, float | None] = {}
    for token, verdict in label_tokens.items():
        logprob = top_logprobs.get(token)
        if logprob is None:
            logprob = top_logprobs.get(f" {token}")
        log_probs[verdict] = logprob

    present = [logprob for logprob in log_probs.values() if logprob is not None]
    if not present:
        raise LabelTokenMissingError(
            f"None of the label tokens {list(label_tokens)!r} appear in the returned "
            f"top_logprobs {sorted(top_logprobs)!r}; refusing to fabricate a distribution."
        )

    # Subtracting the maximum keeps exp() in range; the distribution is invariant to it.
    max_logprob = max(present)
    weights = [math.exp(logprob - max_logprob) for logprob in present]
    total = sum(weights)
    distribution: dict[ComplianceVerdict, float] = {}
    weight_iter = iter(weights)
    for verdict, logprob in log_probs.items():
        distribution[verdict] = next(weight_iter) / total if logprob is not None else 0.0
    return distribution


def _report_absent_labels(
    distribution: Mapping[ComplianceVerdict, float],
    label_tokens: Mapping[str, ComplianceVerdict],
    *,
    top_logprobs: int,
    diagnostics: list[str] | None,
    stage: str,
) -> None:
    """Warn about label tokens that the top-K cut removed from the distribution."""
    absent = [
        f"{token!r} ({verdict.value})"
        for token, verdict in label_tokens.items()
        if distribution.get(verdict, 0.0) == 0.0
    ]
    if not absent:
        return
    record_diagnostic(
        diagnostics,
        f"{stage}: label token(s) {', '.join(absent)} absent from the returned "
        f"top-{top_logprobs} candidates; assigned probability 0.0 and renormalised over the "
        "remaining labels.",
        stacklevel=3,
    )


def label_distribution(
    client: ChatClient,
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    max_retry: int = 3,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    label_tokens: Mapping[str, ComplianceVerdict] = TOKEN_TO_VERDICT,
    diagnostics: list[str] | None = None,
    stage: str = "label elicitation",
    extra_body: Mapping[str, Any] | None = None,
) -> dict[ComplianceVerdict, float]:
    """Ask for one label token and return its restricted-softmax distribution.

    A single chat-completion call with `max_tokens=LABEL_TOKEN_MAX_TOKENS` and logprobs is made
    per attempt. Transport and endpoint failures are retried with exponential backoff, bounded
    by `max_retry`. Deterministic failures (no logprobs, no label token in the candidates) are
    raised immediately: repeating a deterministic request cannot change the answer and would
    only spend calls.

    A label that is present in the return value with probability 0.0 was absent from the top-K
    candidates. That condition is reported as a `TFBearWarning` and appended to `diagnostics`,
    because the renormalisation changes the distribution.

    Args:
        client: Any OpenAI-compatible client exposing the `ChatClient` shape.
        model: Model identifier passed to the endpoint.
        messages: Chat messages of the prompt; the model must answer with one digit.
        max_retry: Maximum number of attempts; must be at least 1.
        temperature: Sampling temperature of the elicitation call; defaults to 0.0 (greedy).
        top_p: Nucleus threshold of the elicitation call; defaults to 0.95.
        top_logprobs: Number of candidate tokens requested per position; defaults to 5.
        label_tokens: Token -> verdict mapping; defaults to the `0`/`1`/`2` digits of
            `TOKEN_TO_VERDICT`.
        diagnostics: Optional run-level collector that receives the absent-label message.
        stage: Stage name used in the absent-label diagnostic.
        extra_body: Optional provider-specific request-body fields (for example
            `{"chat_template_kwargs": {"enable_thinking": False}}`). Omitted when empty.

    Returns:
        A mapping over all `ComplianceVerdict` members. Labels absent from the returned
        candidates have probability 0.0; the present ones are renormalised and sum to 1.0.

    Raises:
        ValueError: If `max_retry` is below 1.
        MissingLogprobsError: If the endpoint returns no logprobs despite the request.
        LabelTokenMissingError: If no label token appears among the returned candidates.
        LogprobElicitationError: If every attempt fails for another reason.
    """
    if max_retry < 1:
        raise ValueError(f"max_retry must be at least 1, got {max_retry}.")

    request_kwargs: dict[str, Any] = {}
    if extra_body:
        request_kwargs["extra_body"] = dict(extra_body)

    last_error: Exception | None = None
    for attempt in range(1, max_retry + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=list(messages),
                max_tokens=LABEL_TOKEN_MAX_TOKENS,
                logprobs=True,
                top_logprobs=top_logprobs,
                temperature=temperature,
                top_p=top_p,
                **request_kwargs,
            )
            distribution = _restricted_softmax(
                _top_logprobs_by_token(response.choices[0].logprobs), label_tokens
            )
            _report_absent_labels(
                distribution,
                label_tokens,
                top_logprobs=top_logprobs,
                diagnostics=diagnostics,
                stage=stage,
            )
            return distribution
        except LogprobElicitationError:
            raise
        except Exception as exc:  # Retried below, then re-raised with attempt context.
            last_error = exc
            if attempt < max_retry:
                time.sleep(2**attempt)

    raise LogprobElicitationError(
        f"Label distribution request failed after {max_retry} attempt(s): {last_error}"
    ) from last_error
