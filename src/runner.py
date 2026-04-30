"""
runner.py — Main entry point for the Model Tracker.

Orchestrates fetching, classification, summarisation, and digest generation.
Saves state (last run timestamp) to avoid re-processing old models.

Usage:
    python src/runner.py [--dry-run] [--config CONFIG] [--output-dir DIR] [--days N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

# Make src importable when running directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fetcher import fetch_new_models, fetch_modelscope_models
from classifier import classify_models
from digest import build_digests
from benchmarks import get_benchmark_scores, warm_caches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("runner")

DEFAULT_CONFIG = ROOT / "config" / "profiles.yaml"
DEFAULT_STATE = ROOT / "state" / "last_run.json"
DEFAULT_OUTPUT = ROOT / "digests"
DEFAULT_TEMPLATES = ROOT / "templates"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read state file %s: %s", state_path, e)
    return {}


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    logger.info("State saved to %s", state_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model Tracker — monitors HuggingFace for new AI models and generates profile digests."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and classify models, print summary to stdout, but do NOT write digest files or update state.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to profiles YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory for digest output files (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE,
        help=f"Path to state JSON file (default: {DEFAULT_STATE})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Override lookback window (days). Ignores state file if set.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Load config ---
    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        return 1
    cfg = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    # --- Determine time window ---
    now = datetime.now(timezone.utc)

    if args.days is not None:
        since = now - timedelta(days=args.days)
        logger.info("Using --days=%d override; looking back since %s", args.days, since.isoformat())
    else:
        # Rolling 24-hour window so counts don't collapse to 0 right after midnight
        # or when the workflow is run more than once per day.
        since = now - timedelta(hours=24)
        logger.info("Using rolling 24-hour lookback since %s", since.isoformat())

    # --- Fetch models ---
    fetcher_cfg = cfg.get("fetcher", {})
    max_models = fetcher_cfg.get("max_models_per_run", 200)
    hf_limit = fetcher_cfg.get("hf_api_limit", 100)

    logger.info("Fetching new models since %s …", since.strftime("%Y-%m-%d %H:%M UTC"))
    try:
        models = fetch_new_models(since=since, max_models=max_models, hf_api_limit=hf_limit)
    except Exception as e:
        logger.error("Failed to fetch models: %s", e)
        return 1

    # --- Fetch ModelScope models (best effort) ---
    modelscope_cfg = cfg.get("modelscope", {}) or {}
    if modelscope_cfg.get("enabled", True):
        ms_max = modelscope_cfg.get("max_models_per_run", max_models)
        try:
            ms_models = fetch_modelscope_models(since=since, max_models=ms_max)
            seen_ids = {m["id"] for m in models}
            added = 0
            for m in ms_models:
                if m["id"] not in seen_ids:
                    models.append(m)
                    seen_ids.add(m["id"])
                    added += 1
            logger.info("Added %d unique ModelScope models (skipped %d duplicates).",
                        added, len(ms_models) - added)
        except Exception as e:
            logger.warning("ModelScope fetch failed (non-fatal): %s", e)

    if not models:
        logger.info("No new models found since last run.")
        if not args.dry_run:
            save_state(args.state_file, {"last_run_at": now.isoformat(), "models_found": 0})
        return 0

    logger.info("Total models fetched: %d", len(models))

    # --- Enrich with benchmark scores (best effort) ---
    hf_token = os.environ.get("HF_TOKEN")
    logger.info("Pre-loading benchmark leaderboard caches …")
    try:
        warm_caches(hf_token=hf_token)
    except Exception as wc_exc:
        logger.debug("warm_caches failed (non-fatal): %s", wc_exc)

    logger.info("Enriching models with benchmark scores …")
    for model in models:
        # Benchmarks come from HuggingFace leaderboards; skip non-HF models.
        if model.get("source") != "huggingface":
            continue
        model_id = model.get("id", "")
        if model_id:
            try:
                scores = get_benchmark_scores(model_id, hf_token=hf_token)
                if scores:
                    model["benchmark_scores"] = scores
            except Exception as bench_exc:
                logger.debug("Benchmark fetch skipped for %s: %s", model_id, bench_exc)

    # --- Classify ---
    classified = classify_models(models, cfg)

    # --- Dry-run output ---
    if args.dry_run:
        _print_dry_run(classified, cfg, since, now)
        return 0

    # --- Write digests ---
    output_cfg = cfg.get("output", {})
    include_json = output_cfg.get("include_json", True)

    written = build_digests(
        classified=classified,
        profiles_cfg=cfg,
        output_dir=args.output_dir,
        run_ts=now,
        include_json=include_json,
        template_dir=DEFAULT_TEMPLATES,
    )

    logger.info("Wrote %d digest file(s):", len(written))
    for p in written:
        logger.info("  → %s", p)

    # --- Save state ---
    save_state(args.state_file, {
        "last_run_at": now.isoformat(),
        "models_found": len(models),
        "digest_files": [str(p) for p in written],
    })

    logger.info("Done. ✅")
    return 0


def _print_dry_run(
    classified: dict[str, list],
    cfg: dict,
    since: datetime,
    now: datetime,
) -> None:
    """Print dry-run summary to stdout."""
    profiles = cfg.get("profiles", {})
    separator = "=" * 72

    print(f"\n{separator}")
    print("  MODEL TRACKER — DRY RUN")
    print(f"  Window: {since.strftime('%Y-%m-%d %H:%M UTC')} → {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(separator)

    total = sum(len(v) for v in classified.values())
    print(f"\nTotal unique models fetched that matched ≥1 profile: {total}\n")

    for profile_key, models in classified.items():
        profile_cfg = profiles.get(profile_key, {})
        display_name = profile_cfg.get("display_name", profile_key)
        commercial_only = profile_cfg.get("commercial_only", False)
        lic_note = "[commercial only]" if commercial_only else "[all licenses]"

        print(f"\n{'─' * 60}")
        print(f"  Profile: {display_name} {lic_note}")
        print(f"  Models matched: {len(models)}")
        print(f"{'─' * 60}")

        if not models:
            print("  (no models matched this profile)")
            continue

        for i, model in enumerate(models[:10], 1):
            mid = model.get("id", "?")
            task = model.get("pipeline_tag") or "N/A"
            lic = model.get("license") or "unknown"
            created = model.get("created_at")
            date_str = created.strftime("%Y-%m-%d") if created and hasattr(created, "strftime") else "?"
            downloads = model.get("downloads") or 0
            reasons = model.get("profile_match_reasons") or []

            print(f"\n  {i:2}. {mid}")
            print(f"      Task: {task} | License: {lic} | Published: {date_str} | Downloads: {downloads:,}")
            print(f"      Match: {'; '.join(reasons)}")

        if len(models) > 10:
            print(f"\n  … and {len(models) - 10} more model(s) not shown.")

    print(f"\n{separator}")
    print("  Dry run complete. No files were written.")
    print(separator + "\n")


if __name__ == "__main__":
    sys.exit(main())
