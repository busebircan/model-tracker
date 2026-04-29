# 🤖 Model Tracker

Automated monitoring of HuggingFace (and ArXiv) for new AI model releases — filtered and summarised per user profile, delivered as daily markdown digests.

---

## What It Does

1. **Fetches** newly released models from HuggingFace Hub (sorted by creation date)
2. **Classifies** each model against 5 user profiles using license checks and keyword matching
3. **Summarises** matched models (task, license, size, downloads, relevance explanation)
4. **Generates** per-profile markdown + JSON digest files in `digests/`
5. **Saves state** (`state/last_run.json`) so each run only processes *new* models

Runs daily via GitHub Actions — digests are committed back to the repo automatically.

---

## User Profiles

| Profile | License Filter | Focus |
|---|---|---|
| **Agent & Tool Use** | Commercial only | Tool-use, code gen, vision, fast inference, embeddings |
| **Agent & Tool Use** | Commercial only | Vision (thermal/IR), offline-capable, RAG, time-series, document understanding |
| **Optimisation & Reasoning** | All licenses | OR/optimization, simulation, code gen (OR-Tools/PuLP), reasoning |
| **Retrieval & Embeddings** | All licenses | Embeddings, rerankers, long-context, chunking |
| **Research & Summarisation** | All licenses | Summarization, research paper analysis, ArXiv monitoring |
| **Safety & Security** | All licenses |  |

Commercial licenses include: `apache-2.0`, `mit`, `bsd-3-clause`, `cc-by-4.0`, `openrail`, `llama3*`, `gemma`, and similar permissive licenses.

---

## Project Structure

```
model-tracker/
├── config/
│   └── profiles.yaml          # Profile definitions, license lists, fetcher settings
├── src/
│   ├── fetcher.py             # HuggingFace Hub API fetching
│   ├── classifier.py          # License + keyword classification
│   ├── summariser.py          # Human-readable model summaries
│   ├── digest.py              # Markdown/JSON digest assembly
│   └── runner.py              # Main entry point (orchestrator)
├── templates/
│   └── digest.md.j2           # Jinja2 markdown template
├── digests/                   # Output digest files (committed by CI)
├── state/
│   └── last_run.json          # State file (last run timestamp)
├── .github/
│   └── workflows/
│       └── run_tracker.yml    # GitHub Actions daily schedule
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

### 4. Look back over a custom window (ignores state)

```bash
python src/runner.py --days 3
```

### 5. Full options

```
python src/runner.py --help

  --dry-run          Fetch and classify, print to stdout, don't write files
  --config PATH      Path to profiles YAML (default: config/profiles.yaml)
  --output-dir DIR   Directory for digest files (default: digests/)
  --state-file PATH  Path to state JSON (default: state/last_run.json)
  --days N           Override lookback window in days
  --verbose / -v     Enable DEBUG logging
```

---

## Configuration (`config/profiles.yaml`)

Edit `config/profiles.yaml` to:

- Add/remove profiles
- Change license allowlists
- Adjust task and tag keywords per profile
- Set `fetcher.lookback_days` (default: 1)
- Set `fetcher.max_models_per_run` (default: 200)

### Commercial License Definition

The config maintains two lists:
- `commercial_licenses` — licenses explicitly allowed for commercial use
- `non_commercial_licenses` — licenses that block commercial use (hard filter for Freya/Thermafy profiles)

Unknown licenses are shown with a ❓ flag and still included (conservative approach — let users judge).

---

## GitHub Actions

The workflow (`.github/workflows/run_tracker.yml`) runs **daily at 07:00 UTC**.

### Setup

1. Fork / clone this repo
2. Go to **Settings → Secrets → Actions**
3. Add `HF_TOKEN` — your HuggingFace read token (optional but avoids rate limits)
4. The workflow will commit digests back to `digests/` and update `state/last_run.json` automatically

### Manual trigger

Go to **Actions → Run Model Tracker → Run workflow** and optionally set:
- `dry_run: true` — preview without committing
- `days: 3` — custom lookback window

---

## Output Example

Each profile gets a digest like `digests/2025-01-15-freya-sub-agents.md`:

```markdown
# Model Tracker Digest — Freya Sub-Agents

**Date:** 2025-01-15
**Profile:** Freya Sub-Agents
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

**Why relevant:** Matched for Freya Sub-Agents via commercial license, task match: text-generation, tag match: code. Capabilities: strong code generation capability.
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

`config/profiles.yaml` includes an `arxiv:` section. The ArXiv fetcher can be enabled to pull recent papers matching your category list and include them in the Personal/Research profile digest.

---

## License

MIT — see [LICENSE](LICENSE)
