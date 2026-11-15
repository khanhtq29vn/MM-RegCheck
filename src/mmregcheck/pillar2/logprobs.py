"""Elicitation of the three-way compliance distribution from label-token logprobs.

Appendix H constrains the model answer to a single digit `y_t in {0, 1, 2}`. One chat call
with `max_tokens=1` and logprobs therefore yields a distribution over the three
`ComplianceVerdict` members directly, without parsing free text.

The draft duplicates this logic in `strategies/prior.py`, `strategies/scoring.py` and
`strategies/utils.py`; this module is the single implementation both stages use, so the prior
and the posterior are elicited identically.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from mmregcheck.pillar2.clients import ChatClient, ChoiceLogprobs
from mmregcheck.schemas import ComplianceVerdict

# Draft benchmark mapping (e.g. `examples/beauty/beauty_tot.py`): the digit the model emits
# maps onto the schema's verdict values in this order. The library defaults of the draft
# ("Refuted"/"Not Enough Info"/"Supported") do not name `ComplianceVerdict` members and are
# deliberately not carried over.
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

    The draft would keep whatever token survived (and renormalise a single remaining label to
    1.0). When no label is present there is nothing to renormalise, so a distribution would
    have to be invented; this error prevents a fabricated Delta downstream.
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

    Draft semantics are kept on purpose: a label outside the top-K candidates gets
    probability 0.0 and the present labels are renormalised, so the result is always a proper
    softmax over what the model actually emitted. Tokens may carry a leading space.
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


def label_distribution(
    client: ChatClient,
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    max_retry: int = 3,
    top_logprobs: int = 5,
    label_tokens: Mapping[str, ComplianceVerdict] = TOKEN_TO_VERDICT,
) -> dict[ComplianceVerdict, float]:
    """Ask for one label token and return its restricted-softmax distribution.

    A single chat-completion call with `max_tokens=1` and logprobs is made per attempt.
    Transport and endpoint failures are retried with the draft's exponential backoff, bounded
    by `max_retry`. Deterministic failures (no logprobs, no label token in the candidates) are
    raised immediately: the request is greedy, so repeating it cannot change the answer and
    would only spend calls.

    Args:
        client: Any OpenAI-compatible client exposing the `ChatClient` shape.
        model: Model identifier passed to the endpoint.
        messages: Chat messages of the prompt; the model must answer with one digit.
        max_retry: Maximum number of attempts; must be at least 1.
        top_logprobs: Number of candidate tokens requested per position.
        label_tokens: Token -> verdict mapping; defaults to the paper's digits.

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

    last_error: Exception | None = None
    for attempt in range(1, max_retry + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=list(messages),
                max_tokens=1,
                logprobs=True,
                top_logprobs=top_logprobs,
                temperature=0.0,
                top_p=1.0,
            )
            return _restricted_softmax(
                _top_logprobs_by_token(response.choices[0].logprobs), label_tokens
            )
        except LogprobElicitationError:
            raise
        except Exception as exc:  # Retried below, then re-raised with attempt context.
            last_error = exc
            if attempt < max_retry:
                time.sleep(2**attempt)

    raise LogprobElicitationError(
        f"Label distribution request failed after {max_retry} attempt(s): {last_error}"
    ) from last_error
