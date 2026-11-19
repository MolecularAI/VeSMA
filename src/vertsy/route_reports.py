"""Pure, template-driven Markdown rendering of one route's forward steps.

Rendering uses Jinja2 from the optional ``vesma[paper]`` dependencies.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from .routes import Route, forward_steps

__all__ = ["render_route_steps"]


def render_route_steps(
    route: Route[str],
    step_template: str,
    metadata_for_step: Callable[[int, Route[str]], Mapping[str, Any]] | None = None,
) -> str:
    """Render each reaction after its precursor-producing steps.

    Only JSON-compatible values reach Jinja. Its sandbox has no file loader, and
    missing template fields fail fast. File I/O and report assembly belong to callers.
    """
    from jinja2 import StrictUndefined
    from jinja2.sandbox import SandboxedEnvironment

    environment = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
    template = environment.from_string(step_template)
    snippets = []
    for index, step in enumerate(forward_steps(route), 1):
        if step.rule is None:
            raise ValueError("forward step has no template ID")
        metadata = metadata_for_step(index, step) if metadata_for_step else {}
        context = json.loads(
            json.dumps(
                {
                    "step_index": index,
                    "product_smiles": step.product,
                    "precursor_smiles": [child.product for child in step.precursors],
                    "template_id": step.rule,
                    "metadata": metadata,
                }
            )
        )
        snippets.append(template.render(context))
    return "\n\n".join(snippets)
