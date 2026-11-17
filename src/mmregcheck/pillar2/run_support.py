"""Shared run loop of the `run-tfbear` command and the batch benchmark script.

Both front ends resolve a per-domain prompt pack, build a `TFBearConfig`, run the samples one at
a time and write two artefacts:

    predictions  one `SubmissionPrediction` per line, the leaderboard submission format
    trace        `<predictions>.trace.jsonl`, one run-settings header line and one JSON object
                 per processed sample (verdict, branches, gate status, usage, diagnostics)

The logic lives here rather than in either front end, so the Typer command and the argparse
script cannot drift apart. Nothing in this module imports a front end, and a key is never
stored, logged or echoed: only the *name* of the API-key environment variable is recorded.

The run loop is designed for long benchmark jobs that must not lose work: predictions are
rewritten atomically after every sample, the trace is flushed after every record, `resume=True`
skips sample ids that are already in the predictions file, and an exception from one sample is
recorded in the trace (without a traceback) while the remaining samples keep running. Every
`TFBearWarning` is shown on stderr, independently of `verbose`.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, cast

from mmregcheck.pillar2.config import TFBearConfig
from mmregcheck.pillar2.diagnostics import TFBearWarning
from mmregcheck.pillar2.engine import StageClients, build_stage_clients, run_tfbear
from mmregcheck.pillar2.llm import EndpointConfig
from mmregcheck.pillar2.pipeline_state import TFBearState
from mmregcheck.pillar2.prompts import (
    ExpansionName,
    PromptPack,
    load_prompt_pack,
)
from mmregcheck.pillar2.results import TFBearResult
from mmregcheck.pillar2.sample_adapter import (
    state_from_sample,
    to_submission_prediction,
    write_predictions,
)
from mmregcheck.schemas import MedRegSampleInput, RegulatoryDomain, SubmissionPrediction

#: Suffix appended to the predictions path to derive the default trace path.
TRACE_SUFFIX = ".trace.jsonl"

#: Version tag of the trace records written by `run_benchmark`.
TRACE_SCHEMA = "tfbear-trace-v1"

#: Shared sampling settings copied from `TFBearConfig` into the trace header.
_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "max_tokens",
    "elicitation_temperature",
    "elicitation_top_p",
    "top_logprobs",
)

#: Per-stage sampling maps copied into the trace header.
_STAGE_SAMPLING_FIELDS = (
    "stage_temperature",
    "stage_top_p",
    "stage_max_tokens",
    "stage_top_logprobs",
)

#: Strategy attributes reported for the resolved expansion stage. A strategy that does not
#: expose one (for example a custom expansion) simply omits it from the trace header.
_EXPANSION_SETTING_NAMES = (
    "n_paths",
    "max_tokens",
    "temperature",
    "top_p",
    "n_steps",
    "k",
    "n_evals",
    "breadth_limit",
    "max_tokens_thought_gen",
    "min_steps",
    "thought_temperature",
    "eval_temperature",
    "thought_top_p",
    "eval_top_p",
)


#: Signature of `warnings.showwarning`; the run replaces it temporarily, never at import time.
_ShowWarning = Callable[[Warning | str, type[Warning], str, int, TextIO | None, str | None], None]


def _show_tfbear_warnings_on_stderr(previous: _ShowWarning) -> _ShowWarning:
    """Return a `warnings.showwarning` that always writes `TFBearWarning` messages to stderr.

    A test runner or a notebook replaces `warnings.showwarning` with its own recorder, which
    would hide a diagnostic from the operator. TF-BEAR records are therefore written to stderr
    directly; every other warning category keeps the handler that was installed before the run.
    """

    def show(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None:
        if issubclass(category, TFBearWarning):
            stream = sys.stderr if file is None else file
            stream.write(warnings.formatwarning(message, category, filename, lineno, line))
        else:
            previous(message, category, filename, lineno, file, line)

    return show


@dataclass(frozen=True, slots=True)
class SamplingSettings:
    """Sampling overrides requested by an operator.

    Every field is a sentinel: `None` means "the operator said nothing", so no value is passed
    to the pack and the pack/`TFBearConfig` default applies. That distinction is why the CLI
    flags have no hardcoded defaults.

    Attributes:
        temperature: Shared temperature of the generation stages (expansion, explanation).
        top_p: Shared nucleus threshold of the generation stages.
        max_tokens: Shared completion budget of the generation stages.
        elicitation_temperature: Shared temperature of the label-token elicitation stages.
        elicitation_top_p: Shared nucleus threshold of the elicitation stages.
        top_logprobs: Candidate cut requested by the elicitation stages.
        verbose: Enables the per-stage `TFBearLogger` output; warnings and diagnostics are shown
            even when it is false.
    """

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    elicitation_temperature: float | None = None
    elicitation_top_p: float | None = None
    top_logprobs: int | None = None
    verbose: bool = False
    extra_body: dict[str, Any] | None = None

    def to_config_kwargs(self) -> dict[str, Any]:
        """Return only the overrides the operator set.

        Omitted fields stay out of the mapping, so `PromptPack.to_config` keeps its own
        defaults for them instead of receiving a value that looks explicit.
        """
        kwargs: dict[str, Any] = {}
        for name in _SAMPLING_FIELDS:
            value = getattr(self, name)
            if value is not None:
                kwargs[name] = value
        if self.verbose:
            kwargs["verbose"] = True
        if self.extra_body:
            kwargs["extra_body"] = dict(self.extra_body)
        return kwargs


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Outcome of one `run_benchmark` invocation.

    Attributes:
        done: Samples that ran successfully in this invocation and were written now.
        failed: Samples that raised; they are recorded in the trace and absent from predictions.
        skipped: Samples skipped because `resume=True` found their id in the predictions file.
        diagnostics: Total number of diagnostics recorded by the successful samples.
        predictions_path: Predictions file the run wrote.
        trace_path: Trace file the run wrote.
    """

    done: int
    failed: int
    skipped: int
    diagnostics: int
    predictions_path: Path
    trace_path: Path


def default_trace_path(predictions_path: Path) -> Path:
    """Return the trace path derived from a predictions path."""
    return predictions_path.with_name(predictions_path.name + TRACE_SUFFIX)


def parse_expansion_param(item: str) -> tuple[str, Any]:
    """Parse one `KEY=VALUE` expansion parameter.

    `VALUE` is read as a boolean (`true`/`false`, case-insensitive), then as an `int`, then as a
    `float`; anything else is rejected instead of being guessed.

    Raises:
        ValueError: If `item` is not `KEY=VALUE` or the value is not a bool, int or float.
    """
    key, separator, raw_value = item.partition("=")
    key = key.strip()
    raw_value = raw_value.strip()
    if not separator or not key or not raw_value:
        raise ValueError(f"invalid --expansion-param {item!r}: expected KEY=VALUE.")

    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return key, lowered == "true"
    try:
        return key, int(raw_value)
    except ValueError:
        pass
    try:
        return key, float(raw_value)
    except ValueError:
        raise ValueError(
            f"invalid value {raw_value!r} for --expansion-param {key!r}: expected an int, "
            "a float or a bool (true/false)."
        ) from None


def parse_extra_body(raw: str | None) -> dict[str, Any] | None:
    """Parse an optional `--extra-body` JSON object string into a dictionary.

    Raises:
        ValueError: If `raw` is not valid JSON or does not decode to a JSON object.
    """
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid --extra-body JSON: {exc.msg}.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"invalid --extra-body: expected a JSON object, got {type(parsed).__name__}."
        )
    return parsed


def parse_expansion_params(items: Sequence[str]) -> dict[str, Any]:
    """Parse repeated `--expansion-param KEY=VALUE` options into one mapping."""
    params: dict[str, Any] = {}
    for item in items:
        key, value = parse_expansion_param(item)
        params[key] = value
    return params


def build_config(
    pack: PromptPack,
    *,
    endpoint: EndpointConfig | None,
    tau_posterior: float,
    tau_delta: float,
    sampling: SamplingSettings,
    expansion_params: Mapping[str, Any] | None = None,
) -> TFBearConfig:
    """Build the `TFBearConfig` of one domain pack, applying only the explicit overrides."""
    kwargs: dict[str, Any] = {
        "endpoint": endpoint,
        "tau_posterior": tau_posterior,
        "tau_delta": tau_delta,
    }
    kwargs.update(sampling.to_config_kwargs())
    if expansion_params:
        kwargs["expansion_params"] = dict(expansion_params)
    return pack.to_config(**kwargs)


def validate_expansion_params(
    expansion: ExpansionName,
    expansion_params: Mapping[str, Any],
) -> None:
    """Reject expansion-parameter keys the chosen variant does not accept, before any sample runs.

    The accepted keys belong to the strategy type, which is the same for every domain pack of a
    given expansion variant, so the check is done on one representative pack. The error message
    comes from `apply_expansion_params` and lists the accepted keys.

    Raises:
        ValueError: If a key is not accepted by the expansion strategy.
    """
    if not expansion_params:
        return
    pack = load_prompt_pack(RegulatoryDomain.BEAUTY, expansion)
    pack.to_config(expansion_params=dict(expansion_params))


def run_one_sample(
    config: TFBearConfig,
    clients: StageClients,
    sample: MedRegSampleInput,
    *,
    close_book: bool,
) -> TFBearResult:
    """Run the pipeline over one benchmark sample and return the full result."""
    state: TFBearState = state_from_sample(sample)
    if close_book:
        state.retrieved_regulatory_context = ""
    return run_tfbear(state, config, clients=clients)


def describe_config(config: TFBearConfig) -> dict[str, Any]:
    """Return the effective settings of one resolved config, without any secret."""
    settings: dict[str, Any] = {
        "sampling": {name: getattr(config, name) for name in _SAMPLING_FIELDS},
        "stage_sampling": {name: dict(getattr(config, name)) for name in _STAGE_SAMPLING_FIELDS},
        "expansion_strategy": {"strategy": type(config.expansion_strategy).__name__},
    }
    for name in _EXPANSION_SETTING_NAMES:
        if hasattr(config.expansion_strategy, name):
            settings["expansion_strategy"][name] = getattr(config.expansion_strategy, name)
    if config.extra_body:
        settings["extra_body"] = dict(config.extra_body)
    return settings


def run_benchmark(
    samples: Sequence[MedRegSampleInput],
    *,
    domain: RegulatoryDomain | None,
    expansion: ExpansionName,
    endpoint: EndpointConfig,
    tau_posterior: float,
    tau_delta: float,
    sampling: SamplingSettings,
    expansion_params: Mapping[str, Any] | None = None,
    close_book: bool,
    predictions_path: Path,
    trace_path: Path,
    resume: bool,
) -> RunSummary:
    """Run TF-BEAR over `samples`, writing predictions and the trace incrementally.

    Every `TFBearWarning` is shown on stderr for the duration of the run, independently of
    `sampling.verbose`, so a diagnostic cannot be lost because an operator forgot a flag.

    Args:
        samples: Benchmark records to run.
        domain: Fixed domain pack, or `None` to use each sample's own domain.
        expansion: Expansion variant of every pack.
        endpoint: Default endpoint of every stage.
        tau_posterior: Posterior threshold of the dual gate.
        tau_delta: Delta threshold of the dual gate.
        sampling: Sampling overrides; omitted fields keep the pack/config defaults.
        expansion_params: Expansion-strategy knobs, already validated and typed.
        close_book: Drop the retrieved regulatory context before the run.
        predictions_path: Leaderboard predictions file.
        trace_path: Trace file; a header line is written at the start of every invocation.
        resume: Skip sample ids already present in `predictions_path` and keep their lines.

    Returns:
        Counts of the invocation and the two artefact paths.

    Raises:
        ValueError: If there are no samples, an expansion parameter is rejected, or an existing
            predictions file cannot be parsed for `resume=True`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("always", TFBearWarning)
        warnings.showwarning = cast(Any, _show_tfbear_warnings_on_stderr(warnings.showwarning))
        return _run_benchmark(
            samples,
            domain=domain,
            expansion=expansion,
            endpoint=endpoint,
            tau_posterior=tau_posterior,
            tau_delta=tau_delta,
            sampling=sampling,
            expansion_params=expansion_params,
            close_book=close_book,
            predictions_path=predictions_path,
            trace_path=trace_path,
            resume=resume,
        )


def _run_benchmark(
    samples: Sequence[MedRegSampleInput],
    *,
    domain: RegulatoryDomain | None,
    expansion: ExpansionName,
    endpoint: EndpointConfig,
    tau_posterior: float,
    tau_delta: float,
    sampling: SamplingSettings,
    expansion_params: Mapping[str, Any] | None,
    close_book: bool,
    predictions_path: Path,
    trace_path: Path,
    resume: bool,
) -> RunSummary:
    if not samples:
        raise ValueError("run_benchmark: no samples to run.")

    sample_domains: list[RegulatoryDomain] = []
    configs: dict[RegulatoryDomain, TFBearConfig] = {}
    for sample in samples:
        sample_domain = domain if domain is not None else sample.domain
        sample_domains.append(sample_domain)
        if sample_domain not in configs:
            pack = load_prompt_pack(sample_domain, expansion)
            configs[sample_domain] = build_config(
                pack,
                endpoint=endpoint,
                tau_posterior=tau_posterior,
                tau_delta=tau_delta,
                sampling=sampling,
                expansion_params=expansion_params,
            )

    # Configs are built eagerly, so a rejected expansion parameter fails before an artefact is
    # touched. Clients are built on the first sample that actually needs one: a fully resumed
    # run must not require an API key for work it is not going to repeat.
    clients: dict[RegulatoryDomain, StageClients] = {}

    existing_predictions = (
        _load_predictions(predictions_path) if resume and predictions_path.is_file() else []
    )
    resumed_sample_ids = {prediction.sample_id for prediction in existing_predictions}
    predictions: list[SubmissionPrediction] = list(existing_predictions)
    if not resume:
        _write_predictions_atomic(predictions_path, predictions)

    header = _header_record(
        domain=domain,
        expansion=expansion,
        endpoint=endpoint,
        tau_posterior=tau_posterior,
        tau_delta=tau_delta,
        verbose=sampling.verbose,
        expansion_params=expansion_params,
        close_book=close_book,
        predictions_path=predictions_path,
        trace_path=trace_path,
        resume=resume,
        configs=configs,
    )

    done = 0
    failed = 0
    skipped = 0
    diagnostics = 0
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a" if resume else "w", encoding="utf-8") as trace_file:
        _write_trace_record(trace_file, header)
        for sample, sample_domain in zip(samples, sample_domains, strict=True):
            if sample.sample_id in resumed_sample_ids:
                skipped += 1
                continue
            if sample_domain not in clients:
                clients[sample_domain] = build_stage_clients(configs[sample_domain])
            try:
                result = run_one_sample(
                    configs[sample_domain],
                    clients[sample_domain],
                    sample,
                    close_book=close_book,
                )
                prediction = to_submission_prediction(result, sample.sample_id)
            except Exception as exc:  # one bad sample must not abort the remaining samples
                failed += 1
                _write_trace_record(
                    trace_file,
                    {
                        "record": "sample",
                        "status": "failed",
                        "sample_id": sample.sample_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                continue

            predictions.append(prediction)
            _write_predictions_atomic(predictions_path, predictions)
            _write_trace_record(
                trace_file,
                {
                    "record": "sample",
                    "status": "ok",
                    "sample_id": sample.sample_id,
                    **result.to_trace_dict(),
                },
            )
            done += 1
            diagnostics += len(result.diagnostics)

    summary = RunSummary(
        done=done,
        failed=failed,
        skipped=skipped,
        diagnostics=diagnostics,
        predictions_path=predictions_path,
        trace_path=trace_path,
    )
    print(
        "TF-BEAR run summary: "
        f"samples_done={done} samples_failed={failed} samples_skipped={skipped} "
        f"diagnostics={diagnostics}",
        file=sys.stderr,
    )
    return summary


def _header_record(
    *,
    domain: RegulatoryDomain | None,
    expansion: ExpansionName,
    endpoint: EndpointConfig,
    tau_posterior: float,
    tau_delta: float,
    verbose: bool,
    expansion_params: Mapping[str, Any] | None,
    close_book: bool,
    predictions_path: Path,
    trace_path: Path,
    resume: bool,
    configs: Mapping[RegulatoryDomain, TFBearConfig],
) -> dict[str, Any]:
    """Build the once-per-invocation trace header from the resolved configs."""
    first = next(iter(configs.values()))
    record: dict[str, Any] = {
        "record": "header",
        "schema": TRACE_SCHEMA,
        "domain": domain.value if domain is not None else "per-sample",
        "expansion": expansion,
        "model": endpoint.model,
        "base_url": endpoint.base_url,
        "api_key_env": "[REDACTED]",
        "tau_posterior": tau_posterior,
        "tau_delta": tau_delta,
        "verbose": verbose,
        "max_retry": first.max_retry,
        "close_book": close_book,
        "resume": resume,
        "predictions_path": str(predictions_path),
        "trace_path": str(trace_path),
        "expansion_params": dict(expansion_params) if expansion_params else None,
        "packs": {
            config_domain.value: describe_config(config)
            for config_domain, config in configs.items()
        },
    }
    record.update(describe_config(first))
    return record


def _write_trace_record(trace_file: TextIO, record: Mapping[str, Any]) -> None:
    """Write one trace JSON object and flush it, so an interrupted run loses no record."""
    trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    trace_file.flush()


def _load_predictions(path: Path) -> list[SubmissionPrediction]:
    """Read a predictions file for `--resume`, reporting the line of a corrupt record."""
    predictions: list[SubmissionPrediction] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            predictions.append(SubmissionPrediction.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: invalid prediction, cannot resume: {exc}") from exc
    return predictions


def _write_predictions_atomic(
    path: Path,
    predictions: Sequence[SubmissionPrediction],
) -> None:
    """Rewrite the whole predictions file through a same-directory temp file and `os.replace`.

    `write_predictions` can only write a full list, so the rewrite is made atomic: readers see
    either the previous complete file or the new complete file, never a partial one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        write_predictions(temporary, predictions)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
