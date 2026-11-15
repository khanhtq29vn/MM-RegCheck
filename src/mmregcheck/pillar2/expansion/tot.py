"""Tree-of-Thoughts reasoning-path expansion.

Breadth-first *exhaustive* search (`bfs_exhaustive`) with LLM-driven early termination and
heuristic pruning: every terminal branch is collected instead of stopping at the first one,
because the expansion stage must hand all surviving paths to scoring. Depth-first search, greedy
BFS and renderings of the tree are deliberately out of scope; this engine only produces paths.

The engine is task-agnostic: the two prompt callables, the heuristic calculator and the response
formats are injected by `ToTExpansionStrategy`. All diagnostics go through `TFBearLogger`, which
is silent unless `verbose=True`.

Model output is parsed as a JSON object. `json-repair` is imported lazily inside
`_parse_json_object` so the library still works without the optional `llm` extra, and unparseable
output raises `ExpansionParseError` instead of silently producing an empty reasoning node.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable
from typing import Any

from mmregcheck.pillar2.base import BaseExpansionStrategy
from mmregcheck.pillar2.llm import create_completion
from mmregcheck.pillar2.logger import TFBearLogger
from mmregcheck.pillar2.pipeline_state import Node, TFBearState, validate_template

#: Value of `Node.meta["strategy"]`; the scoring stage copies it into `TFBearBranchTrace.strategy`.
TOT_STRATEGY = "tot"

# Evaluator label -> numeric value used by the value-style state evaluator.
_EVALUATION_TO_SCORE: dict[str, float] = {"sure": 1.0, "likely": 0.5, "impossible": 0.0}

# Default JSON-mode response_format; callers can override it per instance.
_DEFAULT_JSON_FORMAT: dict[str, Any] = {"type": "json_object"}


class ExpansionParseError(ValueError):
    """Raised when model output cannot be turned into a reasoning step or an evaluation."""


def _parse_json_object(raw: str, *, context: str) -> dict[str, Any]:
    """Parse a JSON object from assistant output, tolerating small syntax slips.

    Plain `json.loads` is tried first because most responses are valid JSON. `json_repair` (the
    optional `llm` extra) is imported lazily only when that fails. Any output that still cannot be
    parsed as an object raises `ExpansionParseError`; nothing is fabricated from garbage.
    """
    text = raw.strip()
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if not isinstance(parsed, dict):
        try:
            import json_repair
        except ImportError as exc:
            raise ExpansionParseError(
                f"{context}: could not parse model output as JSON ({raw[:200]!r}). "
                "Install the optional 'llm' extra for tolerant parsing via json-repair."
            ) from exc
        try:
            parsed = json_repair.loads(text)
        except Exception as exc:
            raise ExpansionParseError(
                f"{context}: could not parse model output as JSON ({raw[:200]!r})."
            ) from exc

    if not isinstance(parsed, dict):
        raise ExpansionParseError(f"{context}: model output is not a JSON object ({raw[:200]!r}).")
    return parsed


def _as_text(value: Any) -> str:
    """Normalise a JSON scalar/container to a non-empty string, or `''`."""
    if value is None:
        return ""
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _evaluation_score(evaluation: dict[str, Any]) -> float:
    """Convert a parsed evaluation object to a score in `[0, 1]`."""
    label = evaluation.get("evaluation", "")
    if isinstance(label, str) and label.strip().lower() in _EVALUATION_TO_SCORE:
        return _EVALUATION_TO_SCORE[label.strip().lower()]
    raw_score = evaluation.get("score")
    if raw_score is not None:
        try:
            return max(0.0, min(1.0, float(raw_score)))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


class TreeNode:
    """One state of the ToT search tree.

    Attributes:
        state: Concatenation of every thought from the root to this node.
        thought: The single thought that produced this node.
        value: Heuristic score assigned by the state evaluator, `None` until evaluated.
        is_terminal: Whether the model declared the problem solved at this node.
        answer: Raw answer emitted by the model when `is_terminal`.
        depth: Number of thoughts on the path; the root has depth 0.
    """

    __slots__ = ("answer", "depth", "is_terminal", "state", "thought", "value")

    def __init__(
        self,
        state: str = "",
        thought: str = "",
        *,
        depth: int = 0,
        value: float | None = None,
        is_terminal: bool = False,
        answer: str | None = None,
    ) -> None:
        self.state = state
        self.thought = thought
        self.value = value
        self.is_terminal = is_terminal
        self.answer = answer
        self.depth = depth


class TreeOfThoughts:
    """Breadth-first ToT search engine over injected prompt callables.

    Only `bfs_exhaustive` is implemented: it collects every terminal node and never stops at the
    first one, which is what the TF-BEAR expansion stage needs (all surviving paths are scored).

    Args:
        client: OpenAI-compatible chat-completion client.
        model: Model identifier passed to the API.
        n_steps: Maximum BFS depth in reasoning steps.
        n_candidates: Candidate thoughts requested per expanded node.
        get_thought_gen_prompt: Maps the current node state to a thought-generation prompt.
        get_state_eval_prompt: Maps the current node state to a state-evaluation prompt.
        n_evals: Independent evaluation calls per state.
        heuristic_calculator: Maps `(state, parsed evaluations)` to a heuristic score.
        breadth_limit: Number of top nodes retained after each non-final BFS level.
        min_steps: BFS levels to expand before honouring `terminate=true`.
        max_retry: Bound on retries per API call.
        max_tokens_thought_gen: Token budget of each thought-generation call.
        thought_response_format: `response_format` for thought generation.
        eval_response_format: `response_format` for state evaluation.
        reasoning: Provider-specific reasoning toggle forwarded to the API.
        logger: Verbose-gated logger; defaults to a silent one.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        n_steps: int,
        n_candidates: int,
        get_thought_gen_prompt: Callable[[str], str],
        get_state_eval_prompt: Callable[[str], str],
        n_evals: int,
        heuristic_calculator: Callable[[str, list[dict[str, Any]]], float],
        breadth_limit: int,
        min_steps: int = 1,
        max_retry: int = 3,
        max_tokens_thought_gen: int = 1000,
        thought_response_format: dict[str, Any] | None = None,
        eval_response_format: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
        logger: TFBearLogger | None = None,
    ) -> None:
        for name, value in (
            ("n_steps", n_steps),
            ("n_candidates", n_candidates),
            ("n_evals", n_evals),
            ("breadth_limit", breadth_limit),
            ("max_retry", max_retry),
        ):
            if value < 1:
                raise ValueError(f"TreeOfThoughts: {name} must be >= 1, got {value}.")

        self.client = client
        self.model = model
        self.root = TreeNode()
        self.n_steps = n_steps
        self.n_candidates = n_candidates
        self.get_thought_gen_prompt = get_thought_gen_prompt
        self.get_state_eval_prompt = get_state_eval_prompt
        self.n_evals = n_evals
        self.heuristic_calculator = heuristic_calculator
        self.breadth_limit = breadth_limit
        self.min_steps = max(1, int(min_steps))
        self.max_retry = max_retry
        self.max_tokens_thought_gen = max_tokens_thought_gen
        self.thought_response_format = thought_response_format
        self.eval_response_format = eval_response_format
        self.reasoning: dict[str, Any] = reasoning if reasoning is not None else {"enabled": False}
        self.logger = logger if logger is not None else TFBearLogger(verbose=False)

    def _chat(self, prompt: str, *, n: int, **kwargs: Any) -> list[str]:
        """Call the model once and return the raw completion strings."""
        response = create_completion(
            self.client,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_retry=self.max_retry,
            logger=self.logger,
            context="TreeOfThoughts",
            n=n,
            temperature=0.7,
            top_p=1.0,
            reasoning=self.reasoning,
            **kwargs,
        )
        return [str(choice.message.content or "") for choice in response.choices]

    def thought_generator(self, state: str) -> tuple[list[str], bool, str | None]:
        """Generate thoughts for `state`.

        Returns:
            The candidate thoughts, whether the model declared the search solved, and the raw
            answer when it did.
        """
        prompt = self.get_thought_gen_prompt(state)
        kwargs: dict[str, Any] = {}
        if self.thought_response_format is not None:
            kwargs["response_format"] = self.thought_response_format

        raw_responses = self._chat(
            prompt, n=self.n_candidates, max_tokens=self.max_tokens_thought_gen, **kwargs
        )

        thoughts: list[str] = []
        is_terminal = False
        answer: str | None = None
        for raw in raw_responses:
            parsed = _parse_json_object(raw, context="thought generation")
            thought = _as_text(parsed.get("thought"))
            terminal = parsed.get("terminate") is True
            if not thought and not terminal:
                raise ExpansionParseError(
                    "thought generation: model output has neither a non-empty 'thought' nor "
                    f"'terminate': true ({raw[:200]!r})."
                )
            if thought:
                thoughts.append(thought)
            if terminal:
                is_terminal = True
                answer = _as_text(parsed.get("answer"))

        return thoughts, is_terminal, answer

    def state_evaluator(self, state: str) -> float:
        """Score one state with `n_evals` independent evaluator calls."""
        prompt = self.get_state_eval_prompt(state)
        kwargs: dict[str, Any] = {}
        if self.eval_response_format is not None:
            kwargs["response_format"] = self.eval_response_format

        raw_evaluations = self._chat(prompt, n=self.n_evals, **kwargs)
        parsed = [_parse_json_object(raw, context="state evaluation") for raw in raw_evaluations]
        value = self.heuristic_calculator(state, parsed)
        self.logger.log_info(f"  State value={value:.4f}", style="dim")
        return value

    def _make_terminal_node(self, node: TreeNode, answer: str | None) -> TreeNode:
        """Build the terminal child state for `node` from the model's final answer."""
        actual_answer = answer if answer and answer.lower() != "none" else ""
        if actual_answer:
            separator = "\n" if node.state else ""
            state = f"{node.state}{separator}Answer: {actual_answer}"
            thought = f"Answer: {actual_answer}"
        else:
            state = node.state.strip()
            thought = "Terminal Node Reached"
        return TreeNode(
            state=state,
            thought=thought,
            depth=node.depth + 1,
            is_terminal=True,
            answer=answer,
        )

    def bfs_exhaustive(self) -> list[TreeNode]:
        """Breadth-first search that collects every terminal node.

        Unlike a greedy BFS, the search never stops at the first terminal node: terminal branches
        are retired and their siblings keep expanding until every active branch terminates, is
        pruned, or reaches `n_steps`. Non-terminal leaves left after the last level are returned as
        fallback candidates with `is_terminal=False`.
        """
        queue: deque[TreeNode] = deque([self.root])
        completed_paths: list[TreeNode] = []

        for step in range(1, self.n_steps + 1):
            if not queue:
                self.logger.log_info(f"Step {step}: queue is empty — search complete.", style="dim")
                break

            next_level_nodes: list[TreeNode] = []
            for _ in range(len(queue)):
                node = queue.popleft()
                self.logger.log_info(
                    f"Step {step}: expanding state {node.state or '<root>'!r}", style="dim"
                )

                thoughts, is_terminal, answer = self.thought_generator(node.state)

                if is_terminal and step >= self.min_steps:
                    terminal_node = self._make_terminal_node(node, answer)
                    if not terminal_node.state:
                        self.logger.log_warning(
                            f"Step {step}: terminal node without state or answer — dropped."
                        )
                        continue
                    completed_paths.append(terminal_node)
                    self.logger.log_info(
                        f"Step {step}: terminal node collected: {terminal_node.state}",
                        style="green",
                    )
                    continue
                if is_terminal:
                    self.logger.log_info(
                        f"Step {step}: terminate=true below min_steps={self.min_steps} "
                        "— expanding anyway.",
                        style="dim",
                    )

                if not thoughts:
                    self.logger.log_warning(f"Step {step}: no valid thoughts — node dropped.")
                    continue

                separator = "\n" if node.state else ""
                for thought in thoughts:
                    next_level_nodes.append(
                        TreeNode(
                            state=f"{node.state}{separator}{thought}",
                            thought=thought,
                            depth=node.depth + 1,
                        )
                    )

            if not next_level_nodes:
                self.logger.log_info("No non-terminal nodes left to expand — search complete.")
                break

            is_final_step = step == self.n_steps
            if not is_final_step:
                for candidate in next_level_nodes:
                    candidate.value = self.state_evaluator(candidate.state)
                next_level_nodes.sort(key=lambda n: n.value or 0.0, reverse=True)
                kept_nodes = next_level_nodes[: self.breadth_limit]
                self.logger.log_info(
                    f"Step {step}: keeping top {len(kept_nodes)} of {len(next_level_nodes)} nodes.",
                    style="dim",
                )
            else:
                # Final-level nodes are never expanded, so scoring them would only burn API calls;
                # the scorer downstream decides which of them survives.
                kept_nodes = next_level_nodes
                self.logger.log_info(
                    f"Final step: collecting {len(kept_nodes)} non-terminal node(s) as fallback.",
                    style="dim",
                )

            queue = deque(kept_nodes)

        if queue:
            remaining_leaves = sorted(queue, key=lambda n: n.value or 0.0, reverse=True)
            self.logger.log_info(
                f"Fallback: returning {len(remaining_leaves)} non-terminal leaf/leaves.",
                style="dim",
            )
            completed_paths.extend(remaining_leaves)

        return completed_paths


class ToTExpansionStrategy(BaseExpansionStrategy):
    """Expand reasoning paths with the Tree-of-Thoughts BFS engine.

    The model must answer the thought-generation prompt with one JSON object per choice:

        {"thought": "<next reasoning step>", "terminate": <bool>, "answer": "<final answer>"}

    and the evaluation prompt with `{"evaluation": "<sure|likely|impossible>"}`; a numeric
    `{"score": <0..1>}` is accepted as a fallback. Output that cannot be parsed raises
    `ExpansionParseError` instead of producing a reasoning node.

    Args:
        n_steps: Maximum BFS depth in reasoning steps.
        k: Candidate thoughts requested per expanded node.
        n_evals: Independent evaluator calls per state.
        breadth_limit: Top-`b` nodes retained after each BFS level.
        prompt_template: Thought-generation prompt; may reference `{text_claim}`,
            `{retrieved_regulatory_context}` and `{state}`.
        eval_prompt_template: Branch-evaluation prompt with the same placeholders as
            `prompt_template`.
        max_tokens_thought_gen: Token budget of each thought-generation call.
        min_steps: BFS levels to expand before honouring `terminate=true`.
        thought_response_format: `response_format` for thought generation; defaults to JSON mode.
        eval_response_format: `response_format` for state evaluation; defaults to JSON mode.
    """

    def __init__(
        self,
        n_steps: int = 3,
        k: int = 3,
        n_evals: int = 2,
        breadth_limit: int = 3,
        prompt_template: str = "",
        eval_prompt_template: str = "",
        max_tokens_thought_gen: int = 1500,
        min_steps: int = 1,
        verbose: bool = False,
        max_retry: int = 3,
        thought_response_format: dict[str, Any] | None = None,
        eval_response_format: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(prompt_template=prompt_template, verbose=verbose, max_retry=max_retry)
        self.n_steps = n_steps
        self.k = k
        self.n_evals = n_evals
        self.breadth_limit = breadth_limit
        self.eval_prompt_template = eval_prompt_template
        if eval_prompt_template:
            validate_template(eval_prompt_template, f"{type(self).__name__}.eval_prompt_template")
        self.max_tokens_thought_gen = max_tokens_thought_gen
        self.min_steps = min_steps
        self.thought_response_format: dict[str, Any] = (
            dict(thought_response_format)
            if thought_response_format is not None
            else dict(_DEFAULT_JSON_FORMAT)
        )
        self.eval_response_format: dict[str, Any] = (
            dict(eval_response_format)
            if eval_response_format is not None
            else dict(_DEFAULT_JSON_FORMAT)
        )

    @staticmethod
    def _heuristic(_state: str, parsed_evaluations: list[dict[str, Any]]) -> float:
        """Average the parsed evaluator labels into one score for a branch."""
        if not parsed_evaluations:
            return 0.0
        scores = [_evaluation_score(evaluation) for evaluation in parsed_evaluations]
        return sum(scores) / len(scores)

    def expand(self, state: TFBearState, client: Any, model: str) -> list[Node]:
        """Return the terminal and best fallback paths found by the BFS engine.

        When the search produced no terminal node at all, its highest-valued fallback leaves are
        promoted to terminal so the scoring stage still receives candidates instead of crashing on
        an empty set.
        """
        engine = TreeOfThoughts(
            client=client,
            model=model,
            n_steps=self.n_steps,
            n_candidates=self.k,
            get_thought_gen_prompt=lambda current: state.render(
                self.prompt_template, state=current
            ),
            get_state_eval_prompt=lambda current: state.render(
                self.eval_prompt_template, state=current
            ),
            n_evals=self.n_evals,
            heuristic_calculator=self._heuristic,
            breadth_limit=self.breadth_limit,
            min_steps=self.min_steps,
            max_retry=self.max_retry,
            max_tokens_thought_gen=self.max_tokens_thought_gen,
            thought_response_format=self.thought_response_format,
            eval_response_format=self.eval_response_format,
            reasoning=self.reasoning,
            logger=self.logger,
        )

        completed = engine.bfs_exhaustive()
        has_terminal = any(tree_node.is_terminal for tree_node in completed)

        nodes: list[Node] = []
        for tree_node in completed:
            nodes.append(
                Node(
                    state=tree_node.state,
                    thought=tree_node.thought,
                    is_terminal=tree_node.is_terminal or not has_terminal,
                    answer=tree_node.answer,
                    value=tree_node.value,
                    meta={"strategy": TOT_STRATEGY, "depth": tree_node.depth},
                )
            )
            self.logger.log_info(
                f"ToT path {len(nodes)} (depth {tree_node.depth}): {tree_node.state}",
                style="dim",
            )
        return nodes
