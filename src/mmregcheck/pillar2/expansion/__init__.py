"""Reasoning-path expansion strategies of the TF-BEAR pipeline (Section 4.2).

    Standard     one direct reasoning path per sampled completion (zero-shot baseline)
    CoT-SC       `n_paths` sampled chain-of-thought paths, aggregated by self-consistency
    ToT          breadth-first tree-of-thoughts search

Every strategy returns `mmregcheck.pillar2.pipeline_state.Node` objects and stores its name in
`Node.meta["strategy"]`, which the scoring stage copies into `TFBearBranchTrace.strategy`.
"""

from mmregcheck.pillar2.expansion.cot_sc import COT_SC_STRATEGY, CoTSCExpansionStrategy
from mmregcheck.pillar2.expansion.standard import STANDARD_STRATEGY, StandardExpansionStrategy
from mmregcheck.pillar2.expansion.tot import TOT_STRATEGY, ExpansionParseError, ToTExpansionStrategy

__all__ = [
    "COT_SC_STRATEGY",
    "STANDARD_STRATEGY",
    "TOT_STRATEGY",
    "CoTSCExpansionStrategy",
    "ExpansionParseError",
    "StandardExpansionStrategy",
    "ToTExpansionStrategy",
]
