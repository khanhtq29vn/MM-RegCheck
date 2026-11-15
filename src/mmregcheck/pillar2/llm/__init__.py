"""LLM plumbing of the TF-BEAR pipeline: clients, logprob elicitation and completions.

`clients` describes and builds OpenAI-compatible clients, `logprobs` turns label-token
log-probabilities into a `ComplianceVerdict` distribution, and `completion` holds the bounded
retry transport shared by the expansion and explanation strategies.

Importing this package stays cheap: the SDK is imported inside `build_client` at call time, so
the core package works without the optional `llm` extra.
"""

from __future__ import annotations

from mmregcheck.pillar2.llm.clients import (
    ChatChoice,
    ChatClient,
    ChatCompletion,
    ChatCompletions,
    ChatMessage,
    ChatResource,
    ChoiceLogprobs,
    EndpointConfig,
    TokenLogprob,
    TopLogprob,
    build_client,
)
from mmregcheck.pillar2.llm.completion import create_completion, with_reasoning
from mmregcheck.pillar2.llm.logprobs import (
    TOKEN_TO_VERDICT,
    LabelTokenMissingError,
    LogprobElicitationError,
    MissingLogprobsError,
    label_distribution,
)

__all__ = [
    "TOKEN_TO_VERDICT",
    "ChatChoice",
    "ChatClient",
    "ChatCompletion",
    "ChatCompletions",
    "ChatMessage",
    "ChatResource",
    "ChoiceLogprobs",
    "EndpointConfig",
    "LabelTokenMissingError",
    "LogprobElicitationError",
    "MissingLogprobsError",
    "TokenLogprob",
    "TopLogprob",
    "build_client",
    "create_completion",
    "label_distribution",
    "with_reasoning",
]
