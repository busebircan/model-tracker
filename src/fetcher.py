"""
fetcher.py — Fetches new models from HuggingFace Hub API and papers from ArXiv.

Filters by creation/submission date, returns structured dicts.
Compatible with huggingface_hub >= 1.x (no ModelFilter, no direction param).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import arxiv
from huggingface_hub import HfApi

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


def fetch_arxiv_papers(
    since: datetime | None = None,
    categories: list[str] | None = None,
    max_results: int = 50,
) -> list[dict[str, Any]]:
    """
    Fetch recent papers from ArXiv matching the given categories.

    Args:
        since: Only return papers submitted after this UTC datetime.
        categories: ArXiv category strings e.g. ["cs.AI", "cs.LG"].
        max_results: Maximum papers to return.

    Returns:
        List of paper dicts with normalised fields.
    """
    if not categories:
        categories = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]

    cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
    logger.info(
        "Fetching ArXiv papers (categories=%s, since=%s, max=%d)…",
        categories,
        since.isoformat() if since else "N/A",
        max_results,
    )

    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=cat_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: list[dict[str, Any]] = []
    try:
        for result in client.results(search):
            published = result.published
            if published and published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            if since and published and published <= since:
                break
            papers.append(_normalise_paper(result))
            if len(papers) >= max_results:
                break
    except Exception as exc:
        logger.error("ArXiv API error: %s", exc)
        raise

    logger.info("Fetched %d papers from ArXiv.", len(papers))
    return papers


def _normalise_paper(result: Any) -> dict[str, Any]:
    """Convert an arxiv.Result object to a plain dict."""
    return {
        "id": result.get_short_id(),
        "title": result.title.strip().replace("\n", " "),
        "authors": [a.name for a in result.authors],
        "abstract": result.summary.strip().replace("\n", " "),
        "published": result.published,
        "categories": list(result.categories),
        "url": result.entry_id,
        "source": "arxiv",
    }


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
