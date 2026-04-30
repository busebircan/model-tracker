"""
digest.py — Assembles per-profile markdown digest reports.

Uses Jinja2 template for structure. Falls back to plain Python string building
if Jinja2 is not available.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from summariser import summarise_model

logger = logging.getLogger(__name__)


def build_digests(
    classified: dict[str, list[dict[str, Any]]],
    profiles_cfg: dict[str, Any],
    output_dir: str | Path,
    run_ts: datetime | None = None,
    include_json: bool = True,
    template_dir: str | Path | None = None,
) -> list[Path]:
    """
    Build one markdown (and optionally JSON) digest file per profile.

    Args:
        classified: Output of classifier.classify_models()
        profiles_cfg: Full YAML config
        output_dir: Where to write digest files
        run_ts: Timestamp for this run (UTC)
        include_json: Also emit a JSON sidecar file
        template_dir: Path to templates/ directory (for Jinja2)

    Returns:
        List of paths of files written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_ts is None:
        run_ts = datetime.now(timezone.utc)

    ts_str = run_ts.strftime("%Y-%m-%dT%H%M%SZ")
    date_str = run_ts.strftime("%Y-%m-%d")

    profiles: dict[str, dict] = profiles_cfg.get("profiles", {})
    written: list[Path] = []

    for profile_key, models in classified.items():
        profile_cfg = profiles.get(profile_key, {})
        display_name = profile_cfg.get("display_name", profile_key)

        # Generate summaries
        summaries = [
            summarise_model(m, profile_key, profile_cfg) for m in models
        ]

        # Build markdown
        md_content = _render_markdown(
            profile_key=profile_key,
            display_name=display_name,
            profile_cfg=profile_cfg,
            models=models,
            summaries=summaries,
            run_ts=run_ts,
            date_str=date_str,
            template_dir=template_dir,
        )

        # Write markdown file
        safe_key = profile_key.replace("_", "-")
        md_path = output_dir / f"latest-{safe_key}.md"
        md_path.write_text(md_content, encoding="utf-8")
        logger.info("Wrote digest: %s (%d models)", md_path, len(models))
        written.append(md_path)

        # Write JSON sidecar
        if include_json:
            json_path = output_dir / f"latest-{safe_key}.json"
            json_data = {
                "profile": profile_key,
                "display_name": display_name,
                "run_at": run_ts.isoformat(),
                "model_count": len(models),
                "models": [_model_to_json(m) for m in models],
            }
            json_path.write_text(
                json.dumps(json_data, indent=2, default=str), encoding="utf-8"
            )
            written.append(json_path)

    return written


def _render_markdown(
    profile_key: str,
    display_name: str,
    profile_cfg: dict[str, Any],
    models: list[dict[str, Any]],
    summaries: list[str],
    run_ts: datetime,
    date_str: str,
    template_dir: str | Path | None,
) -> str:
    """Render the markdown digest, using Jinja2 if available."""
    if JINJA2_AVAILABLE and template_dir and Path(template_dir, "digest.md.j2").exists():
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape([]),
            keep_trailing_newline=True,
        )
        tmpl = env.get_template("digest.md.j2")
        return tmpl.render(
            profile_key=profile_key,
            display_name=display_name,
            profile_cfg=profile_cfg,
            models=models,
            summaries=summaries,
            run_ts=run_ts,
            date_str=date_str,
        )

    # Fallback: plain Python string building
    return _render_fallback(
        display_name=display_name,
        profile_cfg=profile_cfg,
        models=models,
        summaries=summaries,
        run_ts=run_ts,
        date_str=date_str,
    )


def _render_fallback(
    display_name: str,
    profile_cfg: dict[str, Any],
    models: list[dict[str, Any]],
    summaries: list[str],
    run_ts: datetime,
    date_str: str,
) -> str:
    """Plain-Python fallback renderer (no Jinja2 dependency)."""
    description = profile_cfg.get("description", "")
    commercial_only = profile_cfg.get("commercial_only", False)
    license_note = "**License filter:** Commercial use only ✅" if commercial_only else "**License filter:** All licenses (no restriction)"

    header = f"""# Model Tracker Digest — {display_name}

**Date:** {date_str}  
**Run timestamp:** {run_ts.strftime("%Y-%m-%d %H:%M UTC")}  
**Profile:** {display_name}  
**Description:** {description}  
{license_note}  
**New models found:** {len(models)}

---
"""

    if not models:
        body = "_No new models matched this profile in this run._\n"
    else:
        body = "\n\n---\n\n".join(summaries) + "\n"

    footer = f"""
---

*Generated by [model-tracker](https://github.com/busebircan/model-tracker) · {run_ts.strftime("%Y-%m-%d %H:%M UTC")}*
"""
    return header + "\n" + body + footer


def _model_to_json(model: dict[str, Any]) -> dict[str, Any]:
    """Prepare model dict for JSON serialisation (handle datetime etc.)."""
    out = {}
    for k, v in model.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out
