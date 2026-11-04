## Summary of Changes

- **Type of PR:** [ ] Leaderboard Submission (`submissions/*.jsonl`) | [ ] Core Module Refactor (`pillar1` / `pillar2`) | [ ] Docs / Website
- **Related Issue:** Fixes #

## Verification Checklist

- [ ] `uv run ruff check src tests scripts` passes with zero errors.
- [ ] `uv run mypy src` passes under `--strict` mode.
- [ ] `uv run pytest` passes (including `test_data_leak_guard.py` verifying no private data or API keys are committed).
