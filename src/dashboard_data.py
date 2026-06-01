"""
dashboard_data.py — Generates docs/data/latest.json for the GitHub Pages dashboard.

Reads all digest JSON files from the digests/ directory and assembles a clean,
sanitised JSON payload for the static dashboard.

Usage:
    python src/dashboard_data.py [--output-dir docs/data] [--digests-dir digests]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dashboard_data")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "profiles.yaml"
DEFAULT_DIGESTS_DIR = ROOT / "digests"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "data"


# Map profile keys → url-friendly id used in filenames
def _profile_key_to_slug(profile_key: str) -> str:
    return profile_key.replace("_", "-")


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _latest_digest_for_profile(digests_dir: Path, profile_key: str) -> dict | None:
    """Read the rolling digest file for a given profile key."""
    slug = _profile_key_to_slug(profile_key)
    digest_path = digests_dir / f"latest-{slug}.json"
    if not digest_path.exists():
        logger.debug("No digest file found for profile %s (%s)", profile_key, digest_path)
        return None
    logger.info("Using digest file: %s", digest_path.name)
    try:
        with open(digest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to read %s: %s", digest_path, e)
        return None


def _sanitise_model(model: dict) -> dict:
    """Return a clean model dict suitable for the dashboard."""
    created = model.get("created_at") or model.get("last_modified") or ""
    # Normalise date
    if isinstance(created, datetime):
        created = created.isoformat()

    downloads = model.get("downloads") or 0
    likes = model.get("likes") or 0
    params = model.get("params_billions")

    tags = model.get("tags") or []
    # Filter noisy tags (pipeline tags, license: prefixes)
    clean_tags = [
        t for t in tags
        if not t.startswith("license:")
        and not t.startswith("arxiv:")
        and len(t) < 40
    ][:15]

    reasons = model.get("profile_match_reasons") or []

    benchmark_scores = model.get("benchmark_scores") or {}
    # Ensure vs_references is serialisable
    if "vs_references" in benchmark_scores:
        vs_refs = benchmark_scores["vs_references"]
        if not isinstance(vs_refs, dict):
            benchmark_scores = {k: v for k, v in benchmark_scores.items() if k != "vs_references"}

    return {
        "id": model.get("id", ""),
        "name": model.get("name") or model.get("id", "").split("/")[-1],
        "url": model.get("url") or f"https://huggingface.co/{model.get('id', '')}",
        "task": model.get("pipeline_tag") or "unknown",
        "license": model.get("license") or "unknown",
        "published_date": created,
        "downloads": int(downloads),
        "likes": int(likes),
        "params_billions": params,
        "tags": clean_tags,
        "match_reasons": reasons,
        "benchmark_scores": benchmark_scores,
        "description": (model.get("description") or "")[:300],
    }


def _update_history(output_dir: Path, profiles_output: dict, run_at: datetime, max_days: int = 90) -> None:
    """Append this run's per-profile model counts and top tags to history.json."""
    history_path = output_dir / "history.json"
    history: dict = {"runs": []}
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as exc:
            logger.warning("Could not read history file: %s", exc)

    run_summary: dict = {"run_at": run_at.isoformat(), "profiles": {}}
    for profile_id, profile_data in profiles_output.items():
        models = profile_data.get("models", [])
        tag_counts: dict[str, int] = {}
        for m in models:
            for t in (m.get("tags") or []):
                tag_counts[t] = tag_counts.get(t, 0) + 1
        top_tags = sorted(tag_counts, key=lambda t: -tag_counts[t])[:10]
        run_summary["profiles"][profile_id] = {
            "model_count": len(models),
            "top_tags": top_tags,
        }

    runs: list = history.get("runs", [])
    runs.append(run_summary)

    cutoff = (run_at - timedelta(days=max_days)).isoformat()
    runs = [r for r in runs if r.get("run_at", "") >= cutoff]
    history["runs"] = runs

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Updated history.json (%d runs)", len(runs))


def build_dashboard_json(
    config_path: Path = DEFAULT_CONFIG,
    digests_dir: Path = DEFAULT_DIGESTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """
    Build docs/data/latest.json from the most recent digest files.

    Returns the path of the written file.
    """
    cfg = _load_config(config_path)
    profiles_cfg: dict = cfg.get("profiles", {})

    now = datetime.now(timezone.utc)

    # Load pre-classified popular models from cache written by popular.py
    popular_classified: dict[str, list[dict]] = {}
    popular_cache_path = output_dir / "popular_cache.json"
    if popular_cache_path.exists():
        try:
            with open(popular_cache_path, "r", encoding="utf-8") as f:
                popular_classified = json.load(f)
            total_classified = sum(len(v) for v in popular_classified.values())
            logger.info("Loaded popular cache: %d models across profiles.", total_classified)
        except Exception as exc:
            logger.warning("Could not read popular cache (non-fatal): %s", exc)
    else:
        logger.info("No popular cache found; fetching popular models inline (first run).")
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from fetcher import fetch_popular_models
            from classifier import classify_models
            popular_raw = fetch_popular_models(top_n=200, sort_by="downloads")
            popular_classified = classify_models(popular_raw, cfg)
            total_classified = sum(len(v) for v in popular_classified.values())
            logger.info(
                "Fetched %d popular models inline; %d classified across profiles.",
                len(popular_raw), total_classified,
            )
        except Exception as exc:
            logger.warning("Could not fetch popular models inline (non-fatal): %s", exc)

    output: dict = {
        "generated_at": now.isoformat(),
        "profiles": {},
    }

    for profile_key, profile_cfg in profiles_cfg.items():
        display_name = profile_cfg.get("display_name", profile_key)
        digest = _latest_digest_for_profile(digests_dir, profile_key)

        if digest:
            raw_models = digest.get("models", [])
            run_at = digest.get("run_at", "")
            model_count = digest.get("model_count", len(raw_models))
        else:
            raw_models = []
            run_at = ""
            model_count = 0

        sanitised_models = [_sanitise_model(m) for m in raw_models]

        # Popular models classified for this profile, sorted by downloads desc.
        popular_for_profile = popular_classified.get(profile_key, [])
        popular_for_profile = sorted(
            popular_for_profile,
            key=lambda m: m.get("downloads") or 0,
            reverse=True,
        )
        sanitised_popular = [_sanitise_model(m) for m in popular_for_profile]

        # Build a profile_id safe for JS (snake_case → camelCase not needed; use snake_case)
        profile_id = profile_key  # e.g. "agent_and_tool"

        output["profiles"][profile_id] = {
            "display_name": display_name,
            "description": profile_cfg.get("description", ""),
            "model_count": model_count,
            "run_at": run_at,
            "models": sanitised_models,
            "popular_models": sanitised_popular,
        }
        logger.info(
            "Profile '%s' → %d recent, %d popular (display: %s)",
            profile_id, len(sanitised_models), len(sanitised_popular), display_name,
        )

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "latest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Wrote dashboard data to %s", out_path)
    _update_history(output_dir, output["profiles"], now)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate docs/data/latest.json for the AI Model Radar dashboard.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--digests-dir", type=Path, default=DEFAULT_DIGESTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        out = build_dashboard_json(
            config_path=args.config,
            digests_dir=args.digests_dir,
            output_dir=args.output_dir,
        )
        print(f"Dashboard data written to: {out}")
        return 0
    except Exception as exc:
        logger.error("Failed to build dashboard data: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
