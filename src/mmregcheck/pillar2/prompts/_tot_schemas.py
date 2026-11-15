"""JSON response schemas appended to the Tree-of-Thoughts prompts.

The thought-generation and state-evaluation prompts end with one of these schemas so the model
replies with a JSON object the engine can parse. The schemas are appended after the templates
have been built, because the prompt itself is rendered with `str.format`; the module-level
constants are therefore the format-safe (escaped) forms, while the `_RAW_*` names keep the
readable original.
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

# Escape every brace once, so the schema can be concatenated into a `str.format` template.
THOUGHT_GEN_SCHEMA = _RAW_THOUGHT_GEN_SCHEMA.replace("{", "{{").replace("}", "}}")
STATE_EVAL_SCHEMA = _RAW_STATE_EVAL_SCHEMA.replace("{", "{{").replace("}", "}}")
