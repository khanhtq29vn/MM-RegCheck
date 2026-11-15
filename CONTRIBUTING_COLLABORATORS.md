# Collaborator Guide: Integrating Refactored Modules (`MM-RegCheck`)

Welcome **Hoang-An Vo** and members of the AISIA / VNUHCM-UIT / SMU / NUS research team!

This repository uses a **Contract-First Modular Architecture** so that refactoring the VLM extraction, RAG retrieval, and `TF-BEAR` reasoning code never conflicts with the Project Website (`docs/`) or the Leaderboard Evaluator (`src/mmregcheck/evaluator.py`).

---

## 1. Where to Put Refactored Code

All core data contracts are locked in `src/mmregcheck/schemas.py`. When pushing refactored experimental code, place your modules in these two directories:

1. **`src/mmregcheck/pillar1/` (Pillar 1: Visual-Regulatory Projector — Section 4.1)**
   - `extractor.py`: Implement `VisualRegulatoryExtractor.extract(sample)` to extract `BusinessMetadata` and canonical clinical techniques $E_T = \{t_1, \dots, t_n\}$ guided by the Medical Glossary.
   - `retriever.py`: Implement `StatutoryHybridRetriever.retrieve(techniques, business_info)` for BM25 + Dense Top-$K$ statutory grounding $\hat{C}$.

2. **`src/mmregcheck/pillar2/` (Pillar 2: Evidence Adjudicator via `TF-BEAR` — Section 4.2)**
   - `tf_bear.py`: The mathematical log-odds filter $\Delta = L(P_{\text{posterior}}) - L(P_{\text{prior}})$ and dual-threshold gate ($P_{\text{posterior}} \ge \tau_p, \Delta \ge \tau_\Delta$) are already implemented and tested in `TFBearAdjudicator`. Plug in your VLM `logprobs` elicitation and trajectory expansion (`R1: Strict`, `R2: Permissive`, `R3: Partial`) there.
   - `explainer.py`: `synthesize_overall_explanation(items)` aggregates technique-level verdicts into the overall `FIDES`-compliant rationale. The generative explanation strategies (`GenerativeExplanationStrategy` and `StructuredGenerativeExplanationStrategy`) plug into Stage 5 of the TF-BEAR engine (`config.py` / `engine.py`).

---

## 2. Developer Workflow (macOS / Linux + `uv`)

```bash
# 1. Clone and install dependencies
git clone https://github.com/khanhtq29vn/MM-RegCheck.git
cd MM-RegCheck
uv sync --all-extras

# 2. Create a feature branch for your module
git checkout -b feat/tf-bear-logprobs-engine

# 3. Run quality checks before pushing (must pass 100%)
uv run ruff check src tests scripts
uv run mypy src
uv run pytest

# 4. Push and open a Pull Request
git push -u origin feat/tf-bear-logprobs-engine
```
