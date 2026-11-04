"""Module 2: Knowledge-Anchored Hybrid Statutory Retrieval (Section 4.1).

Hoang-An Vo & Tam Nguyen: Plug the refactored BM25 + Dense Vector Top-K retrieval
logic into `StatutoryRetrieverProtocol` below.
"""

from __future__ import annotations

from typing import Protocol

from mmregcheck.schemas import MedRegSampleInput


class StatutoryRetrieverProtocol(Protocol):
    """Interface contract for Pillar 1 Top-K statutory evidence retrieval."""

    def retrieve_top_k(self, sample: MedRegSampleInput, techniques: list[str], k: int = 5) -> list[str]:
        r"""Return the Top-K statutory provisions $\hat{C}$ for `techniques`."""
        ...
