# 🤖 Model Tracker

Automated monitoring of HuggingFace for new AI model releases — filtered and summarised per user profile, delivered as rolling markdown digests updated every 3 hours.

---

## Dashboard

Live digest → **[busebircan.github.io/model-tracker](https://busebircan.github.io/model-tracker/)**

- **Most Recent** panel: updated every 3 hours via GitHub Actions
- **Most Popular** panel: refreshed daily at 07:00 UTC

---

## What It Does

1. **Fetches** models released in the last 24 hours from HuggingFace Hub (rolling window)
2. **Classifies** each model against 6 use cases using license checks and keyword matching
3. **Enriches** matched models with benchmark scores (HF Open LLM Leaderboard, MTEB)
4. **Summarises** matched models (task, license, size, downloads, relevance explanation)
5. **Generates** per-profile rolling digest files (`digests/latest-<profile>.md/.json`)
6. **Tracks popular models** (top downloads on HuggingFace) separately, refreshed daily

Digests are committed back to the repo automatically after each run.

---

## User Profiles

| Profile | License Filter | Focus |
|---|---|---|
| **Agent & Tool Use** | Commercial only | Tool-use, code gen, vision, fast inference, embeddings |
| **Vision & Edge Deployment** | Commercial only | Vision (thermal/IR), offline-capable, RAG, time-series, document understanding |
| **Optimisation & Reasoning** | All licenses | OR/optimization, simulation, code gen (OR-Tools/PuLP), reasoning |
| **Retrieval & Embeddings** | All licenses | Embeddings, rerankers, long-context, chunking |
| **Research & Summarisation** | All licenses | Summarization, research paper analysis, ArXiv monitoring |
| **Safety & Security** | All licenses | Guardrails, content filtering, PII detection, prompt injection defence, bias evaluation, alignment classifiers, output validation |

Commercial licenses include: `apache-2.0`, `mit`, `bsd-3-clause`, `cc-by-4.0`, `openrail`, `llama3*`, `gemma`, and similar permissive licenses.

---

## Project Structure

```
model-tracker/
├── config/
│   └── profiles.yaml          # Profile definitions, license lists, fetcher settings
├── src/
│   ├── fetcher.py             # HuggingFace Hub API fetching (recent + popular)
│   ├── classifier.py          # License + keyword classification
│   ├── summariser.py          # Human-readable model summaries
│   ├── digest.py              # Markdown/JSON digest assembly
│   ├── benchmarks.py          # Benchmark score enrichment (HF Leaderboard, MTEB)
│   ├── popular.py             # Popular models fetcher (top downloads, daily)
│   ├── dashboard_data.py      # Generates docs/data/latest.json for the dashboard
│   └── runner.py              # Main entry point (orchestrator)
├── templates/
│   └── digest.md.j2           # Jinja2 markdown template
├── digests/                   # Rolling digest files (latest-<profile>.md/.json)
├── state/
│   └── last_run.json          # State file (last run timestamp)
├── docs/
│   └── data/
│       ├── latest.json        # Dashboard data (recent models, all profiles)
│       └── popular_cache.json # Dashboard data (popular models, all profiles)
├── .github/
│   └── workflows/
│       ├── run_tracker.yml    # Runs every 3 hours — fetches recent models
│       └── refresh_popular.yml # Runs daily at 07:00 UTC — refreshes popular models
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run a dry run (no files written)

```bash
python src/runner.py --dry-run
```

### 3. Run normally (writes digests + updates state)

```bash
python src/runner.py
```

### 4. Look back over a custom window

```bash
python src/runner.py --days 3
```

### 5. Refresh popular models manually

```bash
python src/popular.py
```

### 6. Full options

```
python src/runner.py --help

  --dry-run          Fetch and classify, print to stdout, don't write files
  --config PATH      Path to profiles YAML (default: config/profiles.yaml)
  --output-dir DIR   Directory for digest files (default: digests/)
  --state-file PATH  Path to state JSON (default: state/last_run.json)
  --days N           Override lookback window in days (default: rolling 24h)
  --verbose / -v     Enable DEBUG logging
```

---

## Configuration (`config/profiles.yaml`)

Edit `config/profiles.yaml` to:

- Add/remove profiles
- Change license allowlists
- Adjust task and tag keywords per profile
- Set `fetcher.max_models_per_run` (default: 200)

The runner uses a rolling 24-hour lookback window by default. Pass `--days N` to override.

### Commercial License Definition

The config maintains two lists:
- `commercial_licenses` — licenses explicitly allowed for commercial use
- `non_commercial_licenses` — licenses that block commercial use (hard filter for commercial-only profiles)

Unknown licenses are shown with a ❓ flag and still included (conservative approach — let users judge).

---

## GitHub Actions

Two workflows run automatically:

| Workflow | Schedule | What it does |
|---|---|---|
| `run_tracker.yml` | Every 3 hours | Fetches recent models, writes digests and dashboard data |
| `refresh_popular.yml` | Daily at 07:00 UTC | Fetches top models by downloads, updates popular panel |

### Setup

1. Fork / clone this repo
2. Go to **Settings → Secrets → Actions**
3. Add `HF_TOKEN` — your HuggingFace read token (optional but avoids rate limits)
4. The workflows will commit digests and dashboard data back to the repo automatically

### Manual trigger

Go to **Actions → Run Model Tracker → Run workflow** and optionally set:
- `dry_run: true` — preview without committing
- `days: 3` — custom lookback window

---

## Output Example

Each profile gets a rolling digest at `digests/latest-<profile-slug>.md`:

```markdown
# Model Tracker Digest — Agent & Tool Use

**Date:** 2025-01-15
**Profile:** Agent & Tool Use
**New models found:** 12

---

### [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
**Author:** Qwen
**Task:** text generation / language modelling
**License:** `apache-2.0` ✅ commercial use allowed
**Size:** ~7B (from model name)
**Published:** 2025-01-14
**Popularity:** 45.2K downloads · 312 likes
**Tags:** `code`, `instruct`, `gguf`

**Why relevant:** Matched for Agent & Tool Use via commercial license, task match: text-generation, tag match: code. Capabilities: strong code generation capability.
```

---

## Extending the Tracker

### Add a new profile

In `config/profiles.yaml`, add a new entry under `profiles:`:

```yaml
my_new_profile:
  display_name: "My New Profile"
  commercial_only: false
  description: "Description of what this profile monitors"
  task_keywords:
    - text-generation
  tag_keywords:
    - my-keyword
```

### Add ArXiv monitoring

`config/profiles.yaml` includes an `arxiv:` section. The ArXiv fetcher can be enabled to pull recent papers matching your category list and include them in the Research & Summarisation profile digest.

---

## License

MIT — see [LICENSE](LICENSE)
