"""Module 1: Concept-Guided Clinical Extraction (Section 4.1).

Hoang-An Vo & Tam Nguyen: Plug the refactored VLM prompt & Medical Glossary
extraction logic into `VisualRegulatoryExtractor.extract_techniques()` below.
"""

from __future__ import annotations

from typing import Protocol

from mmregcheck.schemas import MedRegSampleInput


class ClinicalExtractorProtocol(Protocol):
    """Interface contract for Pillar 1 clinical technique extraction."""

    def extract_techniques(self, sample: MedRegSampleInput) -> list[str]:
        """Normalize multimodal cues $(T, I)$ into canonical clinical techniques $E_T$."""
        ...


class ReferenceGlossaryExtractor:
    """Deterministic glossary-grounded reference extractor for CI and local testing."""

    CANONICAL_GLOSSARY: tuple[tuple[str, str], ...] = (
        ("tca peel", "Chemical peeling (TCA)"),
        ("peel md", "Chemical peeling (MD)"),
        ("electroporation", "Treatment using electroporation technology"),
        ("laser", "Treatment using fractional laser machine"),
        ("filler", "Dermal filler injection"),
        ("meso", "Mesotherapy subdermal injection"),
    )

    def extract_techniques(self, sample: MedRegSampleInput) -> list[str]:
        """Extract canonical techniques matched against the medical glossary."""
        lower_text = sample.text_claim.lower()
        found: list[str] = []
        for keyword, canonical in self.CANONICAL_GLOSSARY:
            if keyword in lower_text and canonical not in found:
                found.append(canonical)
        return found or ["General skincare claim"]
