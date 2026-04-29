"""
classifier.py — Classifies models against each user profile.

Applies license filters (hard block for commercial-only profiles) and
keyword matching against tags and pipeline_tag.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def classify_models(
    models: list[dict[str, Any]],
    profiles_cfg: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """
    Classify a list of model dicts against all profiles defined in config.

    Args:
        models: List of normalised model dicts from fetcher.py
        profiles_cfg: Full parsed YAML config dict

    Returns:
        Dict mapping profile_key -> list of matched model dicts
        (each model dict gains a 'profile_match_reasons' key)
    """
    commercial_licenses: set[str] = {
        lic.lower() for lic in profiles_cfg.get("commercial_licenses", [])
    }
    non_commercial_licenses: set[str] = {
        lic.lower() for lic in profiles_cfg.get("non_commercial_licenses", [])
    }
    profiles: dict[str, dict] = profiles_cfg.get("profiles", {})

    results: dict[str, list[dict[str, Any]]] = {key: [] for key in profiles}

    for model in models:
        model_license = (model.get("license") or "").lower()
        is_commercial = _is_commercial(
            model_license, commercial_licenses, non_commercial_licenses
        )
        model["is_commercial"] = is_commercial

        for profile_key, profile in profiles.items():
            match, reasons = _match_profile(model, profile, is_commercial)
            if match:
                enriched = dict(model)
                enriched["profile_match_reasons"] = reasons
                results[profile_key].append(enriched)

    for key, matched in results.items():
        logger.info(
            "Profile '%s': %d models matched", profiles[key].get("display_name", key), len(matched)
        )

    return results


def _is_commercial(
    license_id: str,
    commercial_set: set[str],
    non_commercial_set: set[str],
) -> bool | None:
    """
    Determine if a license is commercially usable.

    Returns:
        True  — definitely commercial
        False — definitely non-commercial
        None  — unknown / unrecognised license
    """
    if not license_id:
        return None  # Unknown

    # Exact match in known sets
    if license_id in commercial_set:
        return True
    if license_id in non_commercial_set:
        return False

    # Partial match heuristics
    lc = license_id.lower()
    if any(kw in lc for kw in ("nc", "non-commercial", "noncommercial", "agpl", "gpl")):
        return False
    if any(kw in lc for kw in ("apache", "mit", "bsd", "openrail", "cc-by-4", "cc-by-s")):
        return True

    return None  # Unknown


def _match_profile(
    model: dict[str, Any],
    profile: dict[str, Any],
    is_commercial: bool | None,
) -> tuple[bool, list[str]]:
    """
    Check if a model matches a profile.

    Returns (matched: bool, reasons: list[str])
    """
    reasons: list[str] = []
    commercial_only: bool = profile.get("commercial_only", False)

    # --- License gate (hard filter) ---
    if commercial_only:
        if is_commercial is False:
            return False, []  # Hard block: non-commercial license
        if is_commercial is True:
            reasons.append("commercial license")
        else:
            # Unknown license — include with a note, but flag it
            reasons.append("license unknown (may not be commercial)")

    # --- Task/pipeline keyword matching ---
    task_keywords: list[str] = [kw.lower() for kw in profile.get("task_keywords", [])]
    pipeline_tag: str = (model.get("pipeline_tag") or "").lower()

    task_match = False
    if task_keywords:
        for kw in task_keywords:
            if kw == pipeline_tag or kw in pipeline_tag:
                reasons.append(f"task match: {pipeline_tag}")
                task_match = True
                break

    # --- Tag keyword matching ---
    tag_keywords: list[str] = [kw.lower() for kw in profile.get("tag_keywords", [])]
    model_tags: list[str] = [t.lower() for t in (model.get("tags") or [])]
    model_name_lower: str = model.get("name", "").lower()
    model_id_lower: str = model.get("id", "").lower()

    tag_match = False
    if tag_keywords:
        matched_tags = []
        for kw in tag_keywords:
            # Check tags list
            if any(kw in tag for tag in model_tags):
                matched_tags.append(kw)
                tag_match = True
            # Also check model name / id for relevance
            elif kw in model_name_lower or kw in model_id_lower:
                matched_tags.append(f"{kw} (name)")
                tag_match = True
        if matched_tags:
            reasons.append(f"tag match: {', '.join(matched_tags[:5])}")

    matched = task_match or tag_match
    return matched, reasons
