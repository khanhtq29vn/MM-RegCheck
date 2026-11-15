"""JSON response schemas appended to the Tree-of-Thoughts prompts.

The draft kept these two constants in `strategies/utils.py` and appended them to the thought
generation and state evaluation prompts after escaping their braces, because the prompt is
rendered with `str.format`. The text is copied verbatim; the module-level constants are the
format-safe (escaped) forms, while the `_RAW_*` names keep the readable original.
"""

from __future__ import annotations

_RAW_THOUGHT_GEN_SCHEMA = """\
Return ONLY a JSON object with the following schema (no markdown fences):
{
  "thought": "<a single proposed next reasoning step description>",
  "terminate": <true if you believe the problem is solved and no more steps are needed, otherwise false>,
  "answer": "<final answer string if terminate is true, otherwise null>"
}
"""

_RAW_STATE_EVAL_SCHEMA = """\
Return ONLY a JSON object with the following schema (no markdown fences):
{
  "evaluation": "<sure | likely | impossible>",
  "reasoning": "<brief explanation>"
}
"""

# Escape every brace once, exactly as the draft did before concatenating the schema into a prompt.
THOUGHT_GEN_SCHEMA = _RAW_THOUGHT_GEN_SCHEMA.replace("{", "{{").replace("}", "}}")
STATE_EVAL_SCHEMA = _RAW_STATE_EVAL_SCHEMA.replace("{", "{{").replace("}", "}}")
