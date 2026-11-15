"""Run-time diagnostics of the TF-BEAR pipeline.

The pipeline keeps two parallel records of every non-fatal condition that changes how a stage
behaves: a Python warning with the dedicated `TFBearWarning` category, so a caller that does not
inspect the result still sees it, and a message appended to the run's diagnostics list, so the
condition survives into `TFBearResult.diagnostics` and can be inspected after the run.

Recording never depends on `verbose`: the verbose logger is for interactive debugging, while
these diagnostics are part of the experimental record.
"""

from __future__ import annotations

import warnings


class TFBearWarning(UserWarning):
    """Warning about a run-time condition that changes how a TF-BEAR stage behaves."""


def record_diagnostic(
    diagnostics: list[str] | None,
    message: str,
    *,
    stacklevel: int = 3,
) -> None:
    """Emit `message` as a `TFBearWarning` and append it to `diagnostics`.

    Args:
        diagnostics: Collector of the run, usually `TFBearState.diagnostics`; `None` keeps only
            the warning, for call sites that have no state (for example a standalone helper).
        message: Human-readable description of the condition and its effect.
        stacklevel: Stack level forwarded to `warnings.warn` so the warning points at the caller
            of the stage rather than at this helper.
    """
    warnings.warn(message, TFBearWarning, stacklevel=stacklevel)
    if diagnostics is not None:
        diagnostics.append(message)
