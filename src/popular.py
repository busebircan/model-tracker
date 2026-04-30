"""
popular.py — Fetches the top HuggingFace models by downloads, classifies them
per profile, and writes docs/data/popular_cache.json.

Run daily (independent of the 3-hourly recent runner).

Usage:
    python src/popular.py [--config CONFIG] [--output-dir docs/data] [--top-n 200]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fetcher import fetch_popular_models
from classifier import classify_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("popular")

DEFAULT_CONFIG = ROOT / "config" / "profiles.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "data"
DEFAULT_TOP_N = 200


def build_popular_cache(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    top_n: int = DEFAULT_TOP_N,
) -> Path:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info("Fetching top %d popular models from HuggingFace …", top_n)
    popular_raw = fetch_popular_models(top_n=top_n, sort_by="downloads")
    logger.info("Fetched %d models.", len(popular_raw))

    classified = classify_models(popular_raw, cfg)
    total = sum(len(v) for v in classified.values())
    logger.info("Classified %d model-profile pairs across %d profiles.", total, len(classified))

    # Sort each profile's models by downloads descending
    for profile_key in classified:
        classified[profile_key] = sorted(
            classified[profile_key],
            key=lambda m: m.get("downloads") or 0,
            reverse=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "popular_cache.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(classified, f, indent=2, default=str)

    logger.info("Wrote popular cache to %s (%d profiles)", out_path, len(classified))
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build popular model cache for the dashboard.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        out = build_popular_cache(
            config_path=args.config,
            output_dir=args.output_dir,
            top_n=args.top_n,
        )
        print(f"Popular cache written to: {out}")
        return 0
    except Exception as exc:
        logger.error("Failed to build popular cache: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
