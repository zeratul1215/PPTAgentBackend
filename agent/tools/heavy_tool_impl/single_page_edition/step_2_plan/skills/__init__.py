"""Content-editing skills for step2.

Explicit registration (no auto-discovery): import each skill module and add its
`SKILL` to the shared `SKILLS` registry. Registration order is just the order
skills are documented to the planner (a presentation choice) — it does NOT imply
an execution order. Cross-skill sequencing is decided per-request by the planner,
guided by each skill's `ordering_note`; the only code-enforced edges are the rare
safety/correctness ones declared via `Skill.hard_before`.

Adding a capability = add a `skills/<id>.py` exposing a module-level `SKILL`
(with its own `ordering_note` and, only if truly needed, `hard_before`), then
register it here.
"""

from __future__ import annotations

from .base import (
    SKILLS,
    Skill,
    SkillResult,
    _build_skill_docs,
    _hard_ordering_edges,
    _skill_ids,
)
from . import (
    image_add,
    image_delete,
    image_replace,
    table_build,
    table_compute,
    table_reshape,
    text_add,
    text_delete,
    text_redact,
    text_rewrite,
    text_translate,
)

for _skill in (
    text_redact.SKILL,
    text_rewrite.SKILL,
    text_translate.SKILL,
    text_add.SKILL,
    text_delete.SKILL,
    table_build.SKILL,
    table_reshape.SKILL,
    table_compute.SKILL,
    image_add.SKILL,
    image_delete.SKILL,
    image_replace.SKILL,
):
    SKILLS[_skill.id] = _skill

__all__ = [
    "SKILLS",
    "Skill",
    "SkillResult",
    "_build_skill_docs",
    "_hard_ordering_edges",
    "_skill_ids",
]
