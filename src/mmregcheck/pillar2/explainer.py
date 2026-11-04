"""Module 4: Explanation Generation & FIDES-Compliant Rationale Synthesis."""

from __future__ import annotations

from mmregcheck.schemas import TechniqueVerdictItem, aggregate_strict_hierarchy


def synthesize_overall_explanation(items: list[TechniqueVerdictItem]) -> tuple[str, str]:
    """Aggregate technique-level verdicts via strict hierarchy and format rationale."""
    overall = aggregate_strict_hierarchy([item.verdict for item in items])
    details = " ".join(f"[{item.technique_name}: {item.verdict.value}] {item.explanation}" for item in items)
    return overall.value, details.strip() or "No techniques extracted."
