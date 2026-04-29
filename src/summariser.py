"""
summariser.py — Generates human-readable summaries for matched models.

Produces structured summary text describing what each model does,
why it's useful for the target profile, key stats, and where to find it.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Friendly task descriptions
TASK_DESCRIPTIONS: dict[str, str] = {
    "text-generation": "text generation / language modelling",
    "text2text-generation": "text-to-text generation (instruction-following / translation)",
    "feature-extraction": "feature extraction / embedding generation",
    "sentence-similarity": "sentence similarity and semantic search",
    "text-ranking": "text ranking and reranking",
    "text-retrieval": "text retrieval for RAG pipelines",
    "image-classification": "image classification",
    "image-to-text": "image captioning / vision-language understanding",
    "image-segmentation": "image segmentation",
    "object-detection": "object detection",
    "visual-question-answering": "visual question answering (VQA)",
    "image-feature-extraction": "image feature extraction / visual embeddings",
    "document-question-answering": "document question answering and understanding",
    "summarization": "document and text summarization",
    "question-answering": "question answering",
    "time-series-forecasting": "time-series forecasting",
    "tabular-regression": "tabular regression",
    "tabular-classification": "tabular classification",
}


def summarise_model(
    model: dict[str, Any],
    profile_key: str,
    profile_cfg: dict[str, Any],
) -> str:
    """
    Generate a markdown summary block for a single model in the context of a profile.

    Returns a multi-line markdown string.
    """
    name = model.get("id", "Unknown")
    url = model.get("url", f"https://huggingface.co/{name}")
    author = model.get("author", "Unknown")
    license_id = model.get("license") or "unspecified"
    pipeline_tag = model.get("pipeline_tag") or "general"
    tags = model.get("tags") or []
    downloads = model.get("downloads") or 0
    likes = model.get("likes") or 0
    params = model.get("params_billions")
    created = model.get("created_at")
    reasons = model.get("profile_match_reasons") or []
    languages = model.get("languages") or []
    is_commercial = model.get("is_commercial")

    # --- Task description ---
    task_desc = TASK_DESCRIPTIONS.get(pipeline_tag, pipeline_tag.replace("-", " "))

    # --- Parameter size line ---
    size_line = ""
    if params:
        size_line = f"**Size:** {params}B parameters  \n"
    else:
        # Try to guess from name
        for size_hint in ["0.5b", "1b", "1.5b", "3b", "7b", "8b", "13b", "14b", "22b", "32b", "70b", "72b", "90b", "110b", "405b"]:
            if size_hint in name.lower():
                size_line = f"**Size:** ~{size_hint.upper()} (from model name)  \n"
                break

    # --- License line ---
    commercial_note = ""
    if is_commercial is True:
        commercial_note = " ✅ commercial use allowed"
    elif is_commercial is False:
        commercial_note = " ⚠️ non-commercial"
    else:
        commercial_note = " ❓ license not confirmed"

    license_line = f"**License:** `{license_id}`{commercial_note}  \n"

    # --- Languages ---
    lang_line = ""
    if languages:
        lang_line = f"**Languages:** {', '.join(languages[:5])}  \n"

    # --- Downloads / popularity ---
    pop_line = f"**Popularity:** {_fmt_number(downloads)} downloads · {_fmt_number(likes)} likes  \n"

    # --- Date ---
    date_line = ""
    if created:
        date_str = created.strftime("%Y-%m-%d") if hasattr(created, "strftime") else str(created)
        date_line = f"**Published:** {date_str}  \n"

    # --- Why it's relevant ---
    relevance = _build_relevance(profile_cfg, tags, pipeline_tag, reasons)

    # --- Notable tags ---
    notable = _notable_tags(tags, limit=8)
    tags_line = f"**Tags:** {notable}  \n" if notable else ""

    summary = f"""\
### [{name}]({url})
**Author:** {author}  
**Task:** {task_desc}  
{license_line}{size_line}{date_line}{lang_line}{pop_line}{tags_line}
**Why relevant:** {relevance}
"""
    return summary.strip()


def _build_relevance(
    profile_cfg: dict[str, Any],
    tags: list[str],
    pipeline_tag: str,
    reasons: list[str],
) -> str:
    """Construct a human-readable relevance sentence."""
    profile_name = profile_cfg.get("display_name", "this profile")
    profile_desc = profile_cfg.get("description", "")

    matched_reasons = ", ".join(reasons) if reasons else "keyword match"
    base = f"Matched for **{profile_name}** via {matched_reasons}."

    # Add a sentence about capability
    tag_lower = [t.lower() for t in tags]
    hints = []
    if "tool-use" in tag_lower or "function-calling" in tag_lower:
        hints.append("supports tool/function calling")
    if "code" in tag_lower or "code-generation" in tag_lower:
        hints.append("strong code generation capability")
    if "vision" in tag_lower or "multimodal" in tag_lower:
        hints.append("multimodal / vision capability")
    if any(kw in tag_lower for kw in ["gguf", "onnx", "quantized", "quantised"]):
        hints.append("available in quantized/offline-friendly formats")
    if "rag" in tag_lower or "retrieval" in tag_lower:
        hints.append("designed for RAG / retrieval use cases")
    if "embedding" in tag_lower or "embeddings" in tag_lower or pipeline_tag in ("feature-extraction", "sentence-similarity"):
        hints.append("produces dense embeddings for semantic search")
    if "reranker" in tag_lower or "reranking" in tag_lower:
        hints.append("cross-encoder reranker for improved retrieval quality")

    if hints:
        base += f" Capabilities: {'; '.join(hints)}."
    return base


def _notable_tags(tags: list[str], limit: int = 8) -> str:
    """Return a comma-separated string of interesting tags, skipping generic ones."""
    skip_prefixes = ("license:", "language:", "region:", "doi:", "base_model:", "arxiv:")
    skip_exact = {"transformers", "pytorch", "safetensors", "en", "dataset:", "model"}
    filtered = []
    for tag in tags:
        lc = tag.lower()
        if any(lc.startswith(p) for p in skip_prefixes):
            continue
        if lc in skip_exact:
            continue
        filtered.append(tag)
    return ", ".join(f"`{t}`" for t in filtered[:limit])


def _fmt_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
