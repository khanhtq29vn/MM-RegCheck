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
   - `tf_bear.py`: The public entry point. `TFBear(config, clients=...)` runs one claim (`run`) or one benchmark sample (`run_sample`, with `open_book=False` for the close-book condition), and `TFBear.from_domain(domain, expansion, ...)` builds the pipeline from a per-domain prompt pack. This module is orchestration only and also re-exports the legacy `TFBearAdjudicator`, `TFBearAdjudicationResult`, `TFBearConfig`, `TFBearResult` and `run_tfbear` names.
   - `adjudication.py`: `TFBearAdjudicator` implements and tests the mathematical log-odds filter $\Delta = L(P_{\text{posterior}}) - L(P_{\text{prior}})$, the dual-threshold gate ($P_{\text{posterior}} \ge \tau_p, \Delta \ge \tau_\Delta$), the majority vote, the summed-posterior tie-break and the Appendix H fallback. `summed_posterior_tally` is the one posterior aggregation shared by the adjudicator and the strategies.
   - `prior.py`, `expansion/` (`standard`, `cot-sc`, `tot`), `scoring.py`: Stages 1–3. The VLM `logprobs` elicitation they share lives in `llm/logprobs.py`; trajectory expansion (`R1: Strict`, `R2: Permissive`, `R3: Partial`) is a strategy you can replace.
   - `explainer.py`: Stage 5. `synthesize_overall_explanation(items)` aggregates technique-level verdicts into the overall `FIDES`-compliant rationale. The generative explanation strategies (`GenerativeExplanationStrategy` and `StructuredGenerativeExplanationStrategy`) plug into the engine here.
   - `base.py`, `config.py`, `pipeline_state.py`, `results.py`, `engine.py`, `tie_break.py`, `diagnostics.py`, `run_support.py`: The `Base*Strategy` contracts, the strategy/endpoint wiring (`TFBearConfig`, including per-stage `stage_endpoints`), the shared `TFBearState`/`Node`, the result and usage dataclasses, and the stage orchestration. `tie_break.py` holds the one tie rule (Non-Compliant > Partly-Compliant > Compliant) shared by every stage, `diagnostics.py` holds `TFBearWarning` and the diagnostics recorded on a result, and `run_support.py` is the helper shared by the CLI and the benchmark script (sampling flags, trace file, resume, failed-sample handling). Subclass a `Base*Strategy` and pass it to `TFBearConfig` to replace a stage.
   - `llm/`: `clients.py` (endpoint configuration and client construction), `logprobs.py` (label-token distributions) and `completion.py` (bounded-retry completion helper, public as `create_completion`, plus `with_reasoning`). `openai` and `json_repair` are optional and imported lazily, so the core package works without the `llm` extra.
   - `prompts/`: The per-domain prompt packs (`beauty.py`, `healthcare.py`).

   The import graph is one-way: `results.py`, `base.py`, `tie_break.py` and `diagnostics.py` sit at the bottom, then `adjudication.py`, `config.py`, `engine.py`, `prompts/`, then `run_support.py` (which `cli.py` and `scripts/run_tfbear_benchmark.py` import), with `tf_bear.py` and `pillar2/__init__.py` on top. Do not import a module from a layer above yours, and do not import underscore-prefixed names across modules.

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
