"""Tests for endpoint configuration and client construction (no network)."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from mmregcheck.pillar2.clients import EndpointConfig, build_client

BASE_URL = "https://endpoint.invalid/v1"
KEY_ENV = "MMREG_TEST_API_KEY"


def _config(**overrides: Any) -> EndpointConfig:
    values: dict[str, Any] = {
        "base_url": BASE_URL,
        "api_key_env": KEY_ENV,
        "model": "test-model",
    }
    values.update(overrides)
    return EndpointConfig(**values)


def test_unset_api_key_environment_variable_fails_early(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)

    with pytest.raises(ValueError, match=KEY_ENV):
        build_client(_config())


def test_empty_api_key_environment_variable_fails_early(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "")

    with pytest.raises(ValueError, match=KEY_ENV):
        build_client(_config())


def test_base_url_is_passed_through_without_a_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "dummy-key")

    client = build_client(_config())

    # Constructing the SDK client is offline; the endpoint must still be the configured one.
    assert client.base_url.host == "endpoint.invalid"
    assert client.base_url.path == "/v1/"


def test_timeout_and_extra_headers_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "dummy-key")
    captured: dict[str, Any] = {}

    def fake_openai(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    import openai

    monkeypatch.setattr(openai, "OpenAI", fake_openai)

    build_client(_config(timeout=12.5, extra_headers={"X-Suite": "tf-bear"}))

    assert captured["base_url"] == BASE_URL
    assert captured["timeout"] == 12.5
    assert captured["default_headers"] == {"X-Suite": "tf-bear"}
    # The key is wired through but never exposed by the factory's own API.
    assert "api_key" in captured


def test_optional_fields_are_omitted_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "dummy-key")
    captured: dict[str, Any] = {}

    def fake_openai(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    import openai

    monkeypatch.setattr(openai, "OpenAI", fake_openai)

    build_client(_config())

    assert "timeout" not in captured
    assert "default_headers" not in captured


def test_missing_optional_dependency_names_the_llm_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "dummy-key")
    monkeypatch.setitem(sys.modules, "openai", None)

    with pytest.raises(ImportError, match=r"mmregcheck\[llm\]"):
        build_client(_config())


@pytest.mark.parametrize(
    ("field", "value"),
    [("base_url", "  "), ("api_key_env", ""), ("model", "")],
)
def test_empty_config_fields_are_rejected(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})
