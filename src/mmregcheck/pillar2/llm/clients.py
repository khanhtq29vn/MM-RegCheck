"""OpenAI-compatible client construction for the TF-BEAR stages.

The pipeline talks to any endpoint that implements the OpenAI chat-completions API; nothing
here names a provider, a model or an API-key variable. A caller describes an endpoint with
`EndpointConfig` (base URL, the *name* of the environment variable that holds the key, model,
and optional timeout/headers) and `build_client` returns a lazily-created SDK client.

The SDK is an optional dependency of the `llm` extra, so the core package keeps importing
without it; only `build_client` needs it at call time.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast


class TopLogprob(Protocol):
    """One candidate token and its log-probability inside `TokenLogprob.top_logprobs`."""

    token: str
    logprob: float


class TokenLogprob(Protocol):
    """Log-probabilities of the token at one generated position."""

    top_logprobs: Sequence[TopLogprob]


class ChoiceLogprobs(Protocol):
    """Log-probability payload of one completion choice."""

    content: Sequence[TokenLogprob] | None


class ChatMessage(Protocol):
    """Message payload of one completion choice."""

    content: str | None


class ChatChoice(Protocol):
    """One completion choice."""

    message: ChatMessage
    logprobs: ChoiceLogprobs | None


class ChatCompletion(Protocol):
    """Response of `client.chat.completions.create`."""

    choices: Sequence[ChatChoice]


class ChatCompletions(Protocol):
    """The `chat.completions` resource of an OpenAI-compatible client."""

    def create(self, **kwargs: Any) -> ChatCompletion:
        """Create a chat completion; keyword arguments follow the OpenAI API."""
        ...


class ChatResource(Protocol):
    """The `chat` resource of an OpenAI-compatible client."""

    completions: ChatCompletions


class ChatClient(Protocol):
    """Duck-typed shape every TF-BEAR stage expects from its `client` argument."""

    chat: ChatResource


@dataclass(frozen=True)
class EndpointConfig:
    """One OpenAI-compatible endpoint.

    Attributes:
        base_url: Root URL of the endpoint, including its version prefix if any.
        env_var: *Name* of the environment variable that holds the credential. The value of
            the variable is never stored on the config, logged or echoed in errors.
        model: Model identifier the endpoint accepts.
        timeout: Optional request timeout in seconds.
        extra_headers: Optional headers sent with every request.
    """

    base_url: str
    env_var: str
    model: str
    timeout: float | None = None
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("EndpointConfig.base_url must not be empty.")
        if not self.env_var.strip():
            raise ValueError("EndpointConfig.env_var must name an environment variable.")
        if not self.model.strip():
            raise ValueError("EndpointConfig.model must not be empty.")


def build_client(config: EndpointConfig) -> ChatClient:
    """Build an OpenAI-compatible client for `config`.

    Raises:
        ValueError: If the environment variable named by `config.env_var` is unset or
            empty. This happens before the SDK is imported so a misconfigured endpoint fails
            immediately and identically for every endpoint.
        ImportError: If the optional SDK is not installed, with the pip command to install it.
    """
    secret = os.environ.get(config.env_var)
    if not secret:
        raise ValueError(
            f"Environment variable {config.env_var!r} is not set. It must hold the API "
            f"key for the endpoint at {config.base_url!r}."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The OpenAI-compatible client requires the optional 'llm' extra: "
            "pip install 'mmregcheck[llm]'."
        ) from exc

    kwargs: dict[str, Any] = {"base_url": config.base_url, "api_key": secret}
    if config.timeout is not None:
        kwargs["timeout"] = config.timeout
    if config.extra_headers:
        kwargs["default_headers"] = dict(config.extra_headers)
    # The SDK client is duck-type compatible with `ChatClient`; the concrete SDK types are
    # far broader than this protocol (typed message payloads, overloaded `create`).
    return cast(ChatClient, OpenAI(**kwargs))
