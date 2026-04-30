"""
fetcher.py — Fetches new models from HuggingFace Hub and ModelScope.

Filters by creation date (since last run), returns structured model dicts
with a unified shape so downstream classification/digest code is source-agnostic.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests
from huggingface_hub import HfApi

MODELSCOPE_ENDPOINT = "https://modelscope.cn/api/v1/dolphin/models"
MODELSCOPE_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

logger = logging.getLogger(__name__)


def fetch_new_models(
    since: datetime | None = None,
    max_models: int = 200,
    hf_api_limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Fetch models created after `since` from HuggingFace Hub.

    Args:
        since: Only return models created after this UTC datetime.
               If None, returns the most recently created models up to max_models.
        max_models: Maximum total models to return.
        hf_api_limit: Page size hint for HF API calls.

    Returns:
        List of model dicts with normalised fields.
    """
    api = HfApi()

    logger.info(
        "Fetching models from HuggingFace Hub (since=%s, limit=%d)…",
        since.isoformat() if since else "N/A",
        max_models,
    )

    try:
        # sort='createdAt' returns newest first
        model_iter = api.list_models(
            sort="createdAt",
            limit=max_models,
            cardData=True,
        )
    except Exception as exc:
        logger.error("HuggingFace API error: %s", exc)
        raise

    models: list[dict[str, Any]] = []
    for info in model_iter:
        created = _parse_date(info.created_at)

        # Stop once we've passed our time window (results are newest-first)
        if since and created and created <= since:
            break

        model_dict = _normalise(info)
        models.append(model_dict)

        if len(models) >= max_models:
            break

    logger.info("Fetched %d models from HuggingFace Hub.", len(models))
    return models


def _normalise(info: Any) -> dict[str, Any]:
    """Convert a HuggingFace ModelInfo object to a plain dict."""
    tags: list[str] = list(info.tags or [])
    pipeline_tag: str | None = info.pipeline_tag

    # card_data and cardData are aliases in different versions
    card = info.card_data or info.cardData

    license_id = _extract_license(card, tags)

    # Downloads / likes
    downloads = getattr(info, "downloads", 0) or 0
    likes = getattr(info, "likes", 0) or 0

    # Parameter count from safetensors metadata when available
    param_count = _extract_params(info)

    model_id: str = info.modelId or ""
    author: str = info.author or (model_id.split("/")[0] if "/" in model_id else "")
    name: str = model_id.split("/")[-1] if "/" in model_id else model_id

    return {
        "id": model_id,
        "author": author,
        "name": name,
        "created_at": _parse_date(info.created_at),
        "last_modified": _parse_date(getattr(info, "lastModified", None) or getattr(info, "last_modified", None)),
        "pipeline_tag": pipeline_tag,
        "tags": tags,
        "license": license_id,
        "downloads": downloads,
        "likes": likes,
        "params_billions": param_count,
        "url": f"https://huggingface.co/{model_id}",
        "description": _extract_description(card),
        "languages": _extract_languages(card),
        "datasets": list(getattr(info, "datasets", None) or []),
        "source": "huggingface",
    }


def fetch_popular_models(
    top_n: int = 50,
    sort_by: str = "downloads",
) -> list[dict[str, Any]]:
    """
    Fetch the most popular models from HuggingFace Hub.

    Args:
        top_n: Number of models to return.
        sort_by: Sort field — 'downloads' (last 30 days) or 'likes'.

    Returns:
        List of normalised model dicts, sorted descending by sort_by.
    """
    api = HfApi()
    logger.info("Fetching top %d models by %s from HuggingFace Hub…", top_n, sort_by)
    try:
        model_iter = api.list_models(sort=sort_by, limit=top_n, cardData=True)
    except Exception as exc:
        logger.error("HuggingFace API error: %s", exc)
        raise

    models: list[dict[str, Any]] = []
    for info in model_iter:
        models.append(_normalise(info))
        if len(models) >= top_n:
            break

    logger.info("Fetched %d popular models.", len(models))
    return models


def fetch_modelscope_models(
    since: datetime | None = None,
    max_models: int = 200,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """
    Fetch newly published models from ModelScope (modelscope.cn).

    Pages through the global discovery endpoint sorted by creation time
    (newest first), stopping once we cross the `since` cutoff or hit `max_models`.

    Args:
        since: Only return models created after this UTC datetime.
        max_models: Hard cap on returned models.
        page_size: Page size for the discovery endpoint (max 100).

    Returns:
        List of normalised model dicts (same shape as HuggingFace fetcher).
    """
    logger.info(
        "Fetching models from ModelScope (since=%s, limit=%d)…",
        since.isoformat() if since else "N/A",
        max_models,
    )

    headers = {
        "User-Agent": MODELSCOPE_USER_AGENT,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    models: list[dict[str, Any]] = []
    page_number = 1
    while len(models) < max_models:
        body = {
            "PageSize": min(page_size, max_models - len(models)),
            "PageNumber": page_number,
            "SortBy": "GmtCreate",
            "Target": "",
            "SingleCriterion": [],
        }
        try:
            resp = requests.put(
                MODELSCOPE_ENDPOINT,
                json=body,
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.error("ModelScope API error on page %d: %s", page_number, exc)
            raise

        data = (payload or {}).get("Data", {}).get("Model", {})
        page_models = data.get("Models") or []
        if not page_models:
            break

        stop = False
        for raw in page_models:
            model = _normalise_modelscope(raw)
            created = model.get("created_at")
            if since and created and created <= since:
                stop = True
                break
            models.append(model)
            if len(models) >= max_models:
                break

        if stop or len(page_models) < body["PageSize"]:
            break
        page_number += 1

    logger.info("Fetched %d models from ModelScope.", len(models))
    return models


def _normalise_modelscope(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a ModelScope API model object to the unified model dict shape."""
    path = raw.get("Path") or ""
    name = raw.get("Name") or ""
    model_id = f"{path}/{name}" if path and name else (name or path)

    tasks = raw.get("Tasks") or []
    pipeline_tag = (tasks[0].get("Name") if tasks else None) or None

    # Combine ModelScope's structured tags with task names so the keyword
    # classifier (which checks tags) works the same as for HuggingFace.
    tags: list[str] = list(raw.get("Tags") or [])
    for t in tasks:
        tname = t.get("Name")
        if tname and tname not in tags:
            tags.append(tname)

    license_id = raw.get("License") or None
    if isinstance(license_id, str):
        license_id = license_id.lower() or None

    org = raw.get("Organization") or {}
    author = org.get("Name") or path

    description = (raw.get("Description") or raw.get("ChineseName") or "").strip() or None

    return {
        "id": model_id,
        "author": author,
        "name": name,
        "created_at": _parse_unix_seconds(raw.get("CreatedTime")),
        "last_modified": _parse_unix_seconds(raw.get("LastUpdatedTime")),
        "pipeline_tag": pipeline_tag,
        "tags": tags,
        "license": license_id,
        "downloads": raw.get("Downloads") or 0,
        "likes": raw.get("Stars") or 0,
        "params_billions": None,
        "url": f"https://modelscope.cn/models/{path}/{name}" if path and name else "https://modelscope.cn/",
        "description": description,
        "languages": [],
        "datasets": [],
        "source": "modelscope",
    }


def _parse_unix_seconds(val: Any) -> datetime | None:
    """ModelScope returns timestamps as Unix epoch seconds."""
    if val is None:
        return None
    try:
        return datetime.fromtimestamp(int(val), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _parse_date(val: Any) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return dt
        except ValueError:
            pass
    return None


def _extract_license(card: Any, tags: list[str]) -> str | None:
    """Extract license identifier from card_data or tags."""
    if card is not None:
        lic = getattr(card, "license", None)
        if lic:
            if isinstance(lic, list):
                return lic[0].lower() if lic else None
            return str(lic).lower()
    # Fallback: look in tags for "license:xxx"
    for tag in tags:
        if tag.lower().startswith("license:"):
            return tag[len("license:"):].lower()
    return None


def _extract_description(card: Any) -> str | None:
    """Try to pull a short description from card_data."""
    if card is None:
        return None
    if hasattr(card, "to_dict"):
        d = card.to_dict()
    elif isinstance(card, dict):
        d = card
    else:
        return None
    return d.get("description") or None


def _extract_languages(card: Any) -> list[str]:
    if card is None:
        return []
    langs = getattr(card, "language", None)
    if langs:
        if isinstance(langs, str):
            return [langs]
        return list(langs)
    return []


def _extract_params(info: Any) -> float | None:
    """Try to estimate param count in billions from safetensors metadata."""
    try:
        safetensors = getattr(info, "safetensors", None)
        if safetensors and hasattr(safetensors, "total"):
            total = safetensors.total
            if total:
                return round(total / 1e9, 2)
    except Exception:
        pass
    return None
