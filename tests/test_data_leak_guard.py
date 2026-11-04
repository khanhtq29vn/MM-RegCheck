"""Automated CI Data-Leak Guard for MedReg-MM.

Ensures no collaborator accidentally commits unencrypted full datasets (>100 KB)
or files missing the anti-contamination canary GUID into `data/`.
"""

from __future__ import annotations

from pathlib import Path

from mmregcheck.schemas import MEDREG_CANARY_GUID


def test_no_unencrypted_large_datasets_in_data_dir() -> None:
    data_dir = Path("data")
    assert data_dir.is_dir()

    for fpath in data_dir.rglob("*"):
        if not fpath.is_file() or fpath.name == "README.md":
            continue
        if fpath.suffix == ".enc":
            continue
        size_bytes = fpath.stat().st_size
        assert size_bytes <= 100_000, (
            f"DATA LEAK GUARD TRIGGERED: Unencrypted file '{fpath}' is {size_bytes:,} bytes "
            f"(>100 KB). Full MedReg-MM splits must be hosted on Gated Hugging Face or encrypted as .enc!"
        )
        if fpath.suffix == ".jsonl":
            text = fpath.read_text(encoding="utf-8")
            assert MEDREG_CANARY_GUID in text, (
                f"Missing anti-contamination canary GUID in '{fpath}'!"
            )
