"""
benchmarks.py — Fetches benchmark scores for a given model.

Sources:
  1. HuggingFace Open LLM Leaderboard (v1 and v2) — loaded once, cached in-process
  2. MTEB Leaderboard API — loaded once, cached in-process
  3. Hardcoded reference baselines for comparison

Design: leaderboard data is loaded ONCE per process (at first call) and cached
in module-level dicts, so bulk enrichment of 200 models is cheap after the
initial fetch. Never raises — returns empty dict on any failure.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference baselines
# ---------------------------------------------------------------------------

REFERENCE_MODELS: dict[str, dict[str, float]] = {
    "meta-llama/Llama-3.1-8B-Instruct":     {"mmlu": 68.1, "arc": 62.1, "hellaswag": 82.1},
    "mistralai/Mistral-7B-Instruct-v0.3":    {"mmlu": 62.5, "arc": 60.0, "hellaswag": 81.0},
    "google/gemma-2-9b-it":                  {"mmlu": 71.3, "arc": 68.4, "hellaswag": 87.5},
    "Qwen/Qwen2.5-7B-Instruct":              {"mmlu": 74.2, "arc": 64.0, "hellaswag": 83.5},
    "microsoft/phi-3-mini-4k-instruct":      {"mmlu": 68.8, "arc": 61.4, "hellaswag": 78.9},
}

# ---------------------------------------------------------------------------
# Module-level caches (loaded once per process)
# ---------------------------------------------------------------------------

_MTEB_CACHE: dict[str, Any] | None = None          # model_name.lower() → entry
_HF_LLM_CACHE: dict[str, dict[str, float]] | None = None  # model_id.lower() → scores

_CACHE_LOADED = {"mteb": False, "hf_llm": False}

# ---------------------------------------------------------------------------
# MTEB leaderboard
# ---------------------------------------------------------------------------

MTEB_API_URL = "https://mteb-leaderboard.hf.space/api/models"


def _load_mteb_data(timeout: int = 10) -> dict[str, Any]:
    """Fetch and cache MTEB leaderboard data keyed by model_name.lower()."""
    global _MTEB_CACHE

    if _CACHE_LOADED["mteb"]:
        return _MTEB_CACHE or {}

    _CACHE_LOADED["mteb"] = True  # mark regardless of success to avoid retry loops

    try:
        resp = requests.get(MTEB_API_URL, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()

        cache: dict[str, Any] = {}
        entries = raw if isinstance(raw, list) else raw.get("models", [])
        for entry in entries:
            name = entry.get("model_name") or entry.get("name") or ""
            if name:
                cache[name.lower()] = entry

        _MTEB_CACHE = cache
        logger.info("Loaded %d MTEB model entries", len(cache))
    except Exception as exc:
        logger.debug("Could not load MTEB data: %s", exc)
        _MTEB_CACHE = {}

    return _MTEB_CACHE or {}


def _get_mteb_scores(model_id: str) -> dict[str, float]:
    """Return MTEB scores for model_id (uses cached data)."""
    try:
        cache = _load_mteb_data()
        if not cache:
            return {}

        key = model_id.lower()
        entry = cache.get(key)

        # Try partial match on the short name
        if entry is None:
            short = model_id.split("/")[-1].lower()
            for k, v in cache.items():
                if short in k:
                    entry = v
                    break

        if entry is None:
            return {}

        result: dict[str, float] = {}

        avg = entry.get("average_score") or entry.get("avg") or entry.get("mean_score")
        if avg is not None:
            try:
                result["mteb_avg"] = round(float(avg), 2)
            except (TypeError, ValueError):
                pass

        task_map = {
            "retrieval":      "mteb_retrieval",
            "sts":            "mteb_sts",
            "classification": "mteb_classification",
            "clustering":     "mteb_clustering",
            "reranking":      "mteb_reranking",
        }
        scores = entry.get("scores") or entry.get("tasks") or {}
        if isinstance(scores, dict):
            for src_key, dst_key in task_map.items():
                val = scores.get(src_key)
                if val is not None:
                    try:
                        result[dst_key] = round(float(val), 2)
                    except (TypeError, ValueError):
                        pass

        return result
    except Exception as exc:
        logger.debug("MTEB lookup failed for %s: %s", model_id, exc)
        return {}


# ---------------------------------------------------------------------------
# HuggingFace Open LLM Leaderboard (batch fetch)
# ---------------------------------------------------------------------------

HF_LLM_V1_DATASET = "open-llm-leaderboard/results"
HF_LLM_V2_DATASET = "open-llm-leaderboard-v2/results"

# Possible column names for each benchmark metric
_V1_SCORE_KEYS: dict[str, list[str]] = {
    "arc":        ["arc", "arc_challenge", "harness|arc:challenge|25"],
    "hellaswag":  ["hellaswag", "harness|hellaswag|10"],
    "mmlu":       ["mmlu", "average_mmlu"],
    "truthfulqa": ["truthfulqa_mc", "truthfulqa", "harness|truthfulqa:mc|0"],
    "winogrande": ["winogrande", "harness|winogrande|5"],
    "gsm8k":      ["gsm8k", "harness|gsm8k|5"],
}

_V2_SCORE_KEYS: dict[str, list[str]] = {
    "ifeval": ["ifeval", "IFEval"],
    "bbh":    ["bbh", "BBH"],
    "math":   ["math", "MATH"],
    "gpqa":   ["gpqa", "GPQA"],
}


def _fetch_leaderboard_rows(dataset: str, hf_token: str | None, timeout: int = 15) -> list[dict]:
    """Fetch up to 100 rows from a HF leaderboard dataset via datasets-server."""
    try:
        params = {
            "dataset": dataset,
            "config": "default",
            "split": "train",
            "offset": 0,
            "length": 100,
        }
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        resp = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params=params,
            headers=headers,
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.debug("datasets-server returned %d for %s", resp.status_code, dataset)
            return []
        return resp.json().get("rows", [])
    except Exception as exc:
        logger.debug("Could not fetch leaderboard rows for %s: %s", dataset, exc)
        return []


def _rows_to_score_index(
    rows: list[dict],
    score_map: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """
    Parse raw rows from datasets-server into:
        { model_id.lower(): {metric: score, ...} }
    """
    index: dict[str, dict[str, float]] = {}
    for wrapper in rows:
        row = wrapper.get("row", {})
        model_name = (
            row.get("model") or row.get("model_name") or row.get("name") or ""
        ).lower().strip()
        if not model_name:
            continue

        scores: dict[str, float] = {}
        for bench_key, possible_cols in score_map.items():
            for col in possible_cols:
                val = row.get(col)
                if val is not None:
                    try:
                        scores[bench_key] = round(float(val), 2)
                        break
                    except (TypeError, ValueError):
                        continue

        if scores:
            index[model_name] = scores

    return index


def _load_hf_llm_cache(hf_token: str | None = None) -> dict[str, dict[str, float]]:
    """
    Load HF LLM leaderboard data once per process.
    Returns {model_id.lower(): {metric: score, ...}}
    """
    global _HF_LLM_CACHE

    if _CACHE_LOADED["hf_llm"]:
        return _HF_LLM_CACHE or {}

    _CACHE_LOADED["hf_llm"] = True

    combined: dict[str, dict[str, float]] = {}

    # v1
    v1_rows = _fetch_leaderboard_rows(HF_LLM_V1_DATASET, hf_token)
    combined.update(_rows_to_score_index(v1_rows, _V1_SCORE_KEYS))
    logger.info("Loaded %d v1 leaderboard entries", len(combined))

    # v2
    v2_rows = _fetch_leaderboard_rows(HF_LLM_V2_DATASET, hf_token)
    v2_index = _rows_to_score_index(v2_rows, _V2_SCORE_KEYS)
    for k, v in v2_index.items():
        combined.setdefault(k, {}).update(v)
    logger.info("Loaded %d total leaderboard entries (v1+v2)", len(combined))

    _HF_LLM_CACHE = combined
    return combined


def _get_hf_leaderboard_scores(model_id: str, hf_token: str | None = None) -> dict[str, float]:
    """Look up leaderboard scores for model_id from the in-process cache."""
    try:
        cache = _load_hf_llm_cache(hf_token)
        if not cache:
            return {}

        key = model_id.lower().strip()
        scores = cache.get(key)

        # Try partial matches
        if scores is None:
            short = model_id.split("/")[-1].lower()
            for k, v in cache.items():
                if short in k or k in key:
                    scores = v
                    break

        return dict(scores) if scores else {}
    except Exception as exc:
        logger.debug("HF leaderboard lookup failed for %s: %s", model_id, exc)
        return {}


# ---------------------------------------------------------------------------
# Reference comparison
# ---------------------------------------------------------------------------

def _compute_vs_references(model_scores: dict[str, float]) -> dict[str, dict[str, float]]:
    """For each reference model, compute delta vs. model_scores for shared metrics."""
    result: dict[str, dict[str, float]] = {}
    for ref_id, ref_scores in REFERENCE_MODELS.items():
        deltas: dict[str, float] = {}
        for metric, ref_val in ref_scores.items():
            model_val = model_scores.get(metric)
            if model_val is not None:
                deltas[metric] = round(model_val - ref_val, 2)
        if deltas:
            result[ref_id] = deltas
    return result


# ---------------------------------------------------------------------------
# Warm up caches (call once to pre-load before bulk enrichment)
# ---------------------------------------------------------------------------

def warm_caches(hf_token: str | None = None) -> None:
    """
    Pre-load leaderboard and MTEB caches. Call this ONCE before bulk enrichment
    to avoid per-model network round-trips. Safe to call multiple times.
    """
    _load_hf_llm_cache(hf_token)
    _load_mteb_data()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_benchmark_scores(model_id: str, hf_token: str | None = None) -> dict:
    """
    Fetch benchmark scores for a given model_id.

    Returns dict with any subset of keys:
        arc, hellaswag, mmlu, truthfulqa, winogrande, gsm8k  (v1 leaderboard)
        ifeval, bbh, math, gpqa                               (v2 leaderboard)
        mteb_avg, mteb_retrieval, mteb_sts, ...               (MTEB)
        vs_references                                         (delta vs baselines)

    Always returns a dict (possibly empty). Never raises.
    """
    if not model_id:
        return {}

    try:
        scores: dict[str, Any] = {}

        # 1. HF leaderboard (uses in-process cache)
        hf_scores = _get_hf_leaderboard_scores(model_id, hf_token)
        scores.update(hf_scores)

        # 2. MTEB (uses in-process cache)
        mteb_scores = _get_mteb_scores(model_id)
        scores.update(mteb_scores)

        # 3. vs_references delta (if we got bench scores)
        bench_keys = {"arc", "hellaswag", "mmlu", "truthfulqa", "gsm8k", "winogrande"}
        model_bench = {k: v for k, v in scores.items() if k in bench_keys}
        if model_bench:
            vs_refs = _compute_vs_references(model_bench)
            if vs_refs:
                scores["vs_references"] = vs_refs

        if scores:
            logger.debug("Benchmark scores for %s: %s", model_id, list(scores.keys()))

        return scores

    except Exception as exc:
        logger.warning("Unexpected error in get_benchmark_scores for %s: %s", model_id, exc)
        return {}
