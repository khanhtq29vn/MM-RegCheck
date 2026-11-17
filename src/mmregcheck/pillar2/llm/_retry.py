"""Shared bounded-retry helpers for LLM stage transport and structured parsing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

_T = TypeVar("_T")

DEFAULT_BACKOFF_CAP_SECONDS = 8.0
RATE_LIMIT_BACKOFF_CAP_SECONDS = 30.0
_NON_RETRYABLE_CLIENT_ERROR_MIN = 400
_NON_RETRYABLE_CLIENT_ERROR_MAX = 499
_RETRYABLE_4XX_STATUS_CODES = frozenset({408, 429})


def extract_status_code(exc: BaseException) -> int | None:
    """Return the HTTP status code attached to an SDK/HTTP exception, if any."""
    direct = getattr(exc, "status_code", None)
    if isinstance(direct, int):
        return direct
    response = getattr(exc, "response", None)
    nested = getattr(response, "status_code", None)
    return nested if isinstance(nested, int) else None


def is_retryable_error(exc: BaseException) -> bool:
    """Return False for deterministic 4xx client errors (except 408/429); True otherwise."""
    status_code = extract_status_code(exc)
    if status_code is None:
        return True
    if _NON_RETRYABLE_CLIENT_ERROR_MIN <= status_code <= _NON_RETRYABLE_CLIENT_ERROR_MAX:
        return status_code in _RETRYABLE_4XX_STATUS_CODES
    return True


def _parse_retry_after_seconds(exc: BaseException) -> float | None:
    """Parse a numeric ``Retry-After`` header from an exception's response, if present."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == "retry-after":
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                return None
            return seconds if seconds >= 0.0 else None
    return None


def compute_backoff_seconds(exc: BaseException, attempt: int) -> float:
    """Compute bounded backoff seconds, honouring ``Retry-After`` and 429 status codes."""
    base_delay = min(2.0**attempt, DEFAULT_BACKOFF_CAP_SECONDS)
    if extract_status_code(exc) == 429:
        retry_after = _parse_retry_after_seconds(exc)
        if retry_after is not None:
            return min(retry_after, RATE_LIMIT_BACKOFF_CAP_SECONDS)
        return min(max(base_delay, 2.0 ** (attempt + 1)), RATE_LIMIT_BACKOFF_CAP_SECONDS)
    return base_delay


def call_with_retry(
    operation: Callable[[], _T],
    *,
    max_retry: int,
    sleep: Callable[[float], None],
    build_error: Callable[[int, Exception | None], Exception],
    on_retryable_error: Callable[[int, Exception], None] | None = None,
    non_retryable_exceptions: tuple[type[BaseException], ...] = (),
) -> _T:
    """Execute ``operation`` up to ``max_retry`` attempts with bounded backoff.

    ``non_retryable_exceptions`` are re-raised as-is immediately. Deterministic 4xx HTTP errors
    (other than 408 and 429) stop immediately and raise ``build_error(attempts_used, last_exc)``.
    """
    last_exc: Exception | None = None
    attempts_used = 0
    for attempt in range(1, max_retry + 1):
        attempts_used = attempt
        try:
            return operation()
        except non_retryable_exceptions:
            raise
        except Exception as exc:
            last_exc = exc
            if on_retryable_error is not None:
                on_retryable_error(attempt, exc)
            if not is_retryable_error(exc):
                break
            if attempt < max_retry:
                sleep(compute_backoff_seconds(exc, attempt))
    raise build_error(attempts_used, last_exc) from last_exc
