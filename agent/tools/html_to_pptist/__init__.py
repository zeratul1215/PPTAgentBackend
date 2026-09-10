"""HTML -> PPTist JSON converter (integrated from experiments/html_to_pptist).

Three layers:

* ``measure``  — Playwright renders step3/step4 flow HTML in a headless browser
  and bakes every visual node down to an absolute-positioned primitive.
* ``mapping``  — pure Python; formats those primitives into PPTist elements and
  normalises every page to a 1000px-wide canvas.
* ``convert``  — orchestration + CLI.

Public API used by the backend deck endpoints:

    from agent_backend.agent.tools.html_to_pptist import convert_html, TARGET_WIDTH

``convert_html(html_path, title=None, assets_dir=None)`` returns the full deck
dict ``{title, width, height, slides:[...]}`` where every top-level ``.page`` in
the HTML becomes one slide. ``assets_dir`` overrides where relative ``<img>``
srcs are resolved from (defaults to the HTML file's own folder); local images
are inlined as data URIs so the result is self-contained.
"""

from __future__ import annotations

from .convert import convert as convert_html
from .mapping import _TARGET_WIDTH as TARGET_WIDTH
from .mapping import build_pptist
from .measure import measure_html
from .pptist_to_understand import slide_to_plan_page, slide_to_understand_input

__all__ = [
    "convert_html",
    "build_pptist",
    "measure_html",
    "TARGET_WIDTH",
    "slide_to_plan_page",
    "slide_to_understand_input",
]
