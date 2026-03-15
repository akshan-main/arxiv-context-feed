# Contextual arXiv Feed

An ingestion system that continuously ingests arXiv papers into a [Contextual AI](https://contextual.ai) Datastore using a multi-stage filtering pipeline.

1. [Daily Pipeline](#1-daily-pipeline) for Daily pipeline <br>
2. [Backfill (Historical Ingestion)](#2-backfill-historical-ingestion) for Historcial pipeline


## How It Works

Papers flow through a 3-stage filter before ingestion:

| Stage | Input | Method | Purpose |
|-------|-------|--------|---------|
| **Stage 1** | Title + RSS snippet | Keyword/phrase matching (stemmed) | Wide net, catches obvious matches |
| **Stage 1.5** | Title + snippet | Discovery Agent (LLM) | Semantic catch; finds papers keywords miss |
| **Stage 2** | Full abstract | Judge Agent (LLM) | Full topicality + quality evaluation |

Acceptance: `quality_i >= 65` (cross-batch validated, 4 batches / 205 papers, F1=77% ± 10%). Judge failures are fail-closed (skip, don't ingest).

---

## 1. Daily Pipeline

Runs automatically via GitHub Actions at 06:00 UTC. Fetches new papers from arXiv RSS, filters through all 3 stages, ingests accepted papers to Contextual AI, and optionally posts top papers to Reddit if Reddit API key and account data are added in secrets.

### Run daily ingestion

```bash
contextual-arxiv-feed run-daily
```

### Dry run (no uploads, no Reddit)

```bash
contextual-arxiv-feed run-daily --dry-run
```

### Shortcut dry-run command

Runs daily or updates pipeline in dry-run mode and generates reports in `artifacts/`.

```bash
contextual-arxiv-feed dry-run --mode daily
```

```bash
contextual-arxiv-feed dry-run --mode updates
```

### Weekly updates

Checks for new paper versions and enriches DOI metadata. Runs automatically on Sundays at 08:00 UTC.

```bash
contextual-arxiv-feed run-updates --lookback-days 7
```

```bash
contextual-arxiv-feed run-updates --lookback-days 7 --dry-run
```

### Citation refresh

Updates citation counts from OpenAlex for papers that have a DOI. Runs automatically on Saturdays at 10:00 UTC.

```bash
contextual-arxiv-feed refresh-citations
```

```bash
contextual-arxiv-feed refresh-citations --dry-run
```

### Prune ChromaDB

Removes old paper chunks from local ChromaDB to free disk space. Default threshold is 270 days (9 months).

```bash
contextual-arxiv-feed prune-chromadb --max-age-days 270
```

```bash
contextual-arxiv-feed prune-chromadb --max-age-days 270 --dry-run
```

### Validate configuration

Checks all YAML config files for errors and reports topic/category mismatches.

```bash
contextual-arxiv-feed validate-config
```

---

## 2. Backfill (Historical Ingestion)

Three modes for ingesting papers from the past. Backfill never posts to Reddit, silent ingestion only. Papers are sorted by citation count (OpenAlex) before processing. If a run approaches the 2h30m time limit, it auto-creates a continuation issue with remaining paper IDs.

### Date range

Searches arXiv for papers updated within the given range (inclusive on both sides). Papers are sorted by citation count (OpenAlex) so the most impactful papers are processed first.

```bash
contextual-arxiv-feed backfill --start 2026-01-01 --end 2026-01-31
```

```bash
contextual-arxiv-feed backfill --start 2026-03-01 --end 2026-03-03 --dry-run
```

#### Top-N selection

Limit to the top N most-cited papers per month or year:

```bash
contextual-arxiv-feed backfill --start 2024-01-01 --end 2024-12-31 --top-n 50 --top-n-granularity month
```

```bash
contextual-arxiv-feed backfill --start 2024-01-01 --end 2024-12-31 --top-n 100 --top-n-granularity year
```

### Single date

```bash
contextual-arxiv-feed backfill-date --date 2026-03-10
```

```bash
contextual-arxiv-feed backfill-date --date 2026-03-10 --top-n 20 --dry-run
```

### Specific papers by identifier

Accepts arXiv IDs, arXiv DOIs, or arXiv URLs. Repeatable `-i` flag.

```bash
contextual-arxiv-feed backfill-identifiers -i 2401.12345
```

```bash
contextual-arxiv-feed backfill-identifiers \
  -i 2401.12345 \
  -i 10.48550/arXiv.1706.03762 \
  -i https://arxiv.org/abs/2401.67890
```

```bash
contextual-arxiv-feed backfill-identifiers -i 2401.12345 --dry-run
```

### Streamlit Backfill Request UI

Lightweight web UI for requesting backfills. Creates a GitHub Issue with a JSON payload; does NOT run ingestion. The [backfill.yml](.github/workflows/backfill.yml) workflow picks up the issue.

```bash
pip install -r streamlit_backfill/requirements.txt
```

```bash
export GITHUB_TOKEN=ghp_...
export GITHUB_REPO=owner/repo
```

```bash
streamlit run streamlit_backfill/app.py
```

For Streamlit Cloud deployment, set `GITHUB_TOKEN` and `GITHUB_REPO` in the app's Secrets settings (Settings -> Secrets).

Features:
- Three modes: Single Date, Date Range, Identifiers (with validation preview)
- Top-N selection: limit to top N papers by citations per month or year
- Creates issue with labels `backfill` + `streamlit-request`
- JSON payload in issue body, parseable by backfill workflow

#### Issue Payload Schema

```json
{
  "request_type": "date_range",
  "date": "",
  "start_date": "2026-03-01",
  "end_date": "2026-03-05",
  "identifiers": [],
  "top_n": 50,
  "top_n_granularity": "month",
  "dry_run": false,
  "note": "Requested historical ingest"
}
```

---

## Installation

```bash
git clone https://github.com/your-org/arxiv-context-feed.git
cd arxiv-context-feed
pip install -e .
```

With Reddit posting support:

```bash
pip install -e ".[reddit]"
```

### Environment Variables

Required for ingestion:

```bash
export CONTEXTUAL_API_KEY="your-contextual-api-key"
export CONTEXTUAL_DATASTORE_ID="your-datastore-id"
```

LLM Judge (Cerebras):

```bash
export LLM_API_KEYS="your-cerebras-key"
export LLM_BASE_URL="https://api.cerebras.ai/v1"
```

See [.env.example](.env.example) for all variables.

---

## Configuration

### Topics ([config/topics.yaml](config/topics.yaml))

6 topic groups covering ~40+ concepts:

| Topic | Key | Covers |
|-------|-----|--------|
| Context Engineering | `context-engineering` | Context windows, poisoning, compression, attention |
| RAG & Retrieval | `rag-retrieval` | RAG, embeddings, vector DB, document parsing, re-ranking |
| LLM Inference | `llm-inference` | Inference optimization, reasoning, chain-of-thought |
| Agents & Tools | `agents-tools` | LLM agents, tool use, multi-agent, agent harness |
| Fine-tuning | `fine-tuning` | Fine-tuning, alignment, prompt engineering, PEFT |
| Gen AI Foundations | `generative-ai-foundations` | Transformers, NLP, language generation |

### Judge ([config/judge.yaml](config/judge.yaml))

3-tier LLM fallback: Cerebras (primary) -> Gemini (secondary) -> local Qwen. Round-robin key rotation across all tiers.

### Reddit ([config/reddit.yaml](config/reddit.yaml))

Posts top daily papers to your subreddit. Daily pipeline only; backfill never posts.

---

## GitHub Actions Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| [daily.yml](.github/workflows/daily.yml) | Daily 06:00 UTC + manual | Full daily pipeline |
| [backfill.yml](.github/workflows/backfill.yml) | Issue (label: `backfill`) + manual | Backfill from dispatch inputs or GitHub Issue |
| [weekly_updates.yml](.github/workflows/weekly_updates.yml) | Sunday 08:00 UTC | Version checks + DOI enrichment |
| [weekly_citations.yml](.github/workflows/weekly_citations.yml) | Saturday 10:00 UTC | Citation count refresh |
| [config_from_issue.yml](.github/workflows/config_from_issue.yml) | Issue with config labels | Validate & create PR |

### Backfill Workflow

Triggers automatically when a GitHub Issue with the `backfill` label is opened (e.g. from the Streamlit app). Can also be triggered manually via workflow dispatch.

- **Issue-based**: opens an issue with `backfill` label containing a JSON payload in the body. The workflow parses the payload and runs the CLI command. After completion, the issue is commented with results and closed.
- **Direct inputs**: mode, date, start_date, end_date, identifiers, top_n, dry_run
- **Auto-continuation**: if a run approaches the 2h30m time limit, it creates a new issue with remaining paper IDs so the workflow auto-triggers a follow-up run.
- **Citation ordering**: papers are sorted by citation count (OpenAlex) before processing so the most impactful papers are ingested first.
- **Top-N selection**: limit ingestion to the top N most-cited papers per month or year.

### Required Secrets

- `CONTEXTUAL_API_KEY`
- `CONTEXTUAL_DATASTORE_ID`
- `LLM_API_KEYS` (primary judge, Cerebras)
- `LLM_BASE_URL` (primary judge endpoint)

### Optional Secrets

- `CONTEXTUAL_BASE_URL` (defaults to `https://api.contextual.ai`)
- `LLM_SECONDARY_API_KEYS`, `LLM_SECONDARY_BASE_URL`, `LLM_SECONDARY_MODEL_ID` (fallback judge, Gemini)
- `OPENALEX_API_KEYS` (citation sorting in backfill)
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USERNAME`, `REDDIT_PASSWORD`

---

## Metadata

### Document Naming

PDFs are named `arxiv:{arxiv_id}v{version}` (e.g., `arxiv:2401.12345v1`).

### custom_metadata (20 fields)

Workspace limit: 35 fields shared across all datastores (arXiv, Reddit, blog). arXiv uses 20 fields. 2KB per-document limit.

```
title, url, arxiv_id, categories, primary_category, authors,
published, source, pdf_url, doi, journal_ref, comments,
topics, quality_verdict, quality_i, confidence_i,
novelty_i, relevance_i, technical_depth_i, citation_count
```

All numeric fields are integers (INT-ONLY enforcement).

---

## Development

```bash
pip install -e ".[dev]"
```

```bash
pytest tests/ -v
```

### Architecture

```
src/contextual_arxiv_feed/
├── cli.py                 # Click CLI (daily + backfill commands)
├── config.py              # Pydantic config models
├── arxiv/                 # arXiv integration (feeds, API, PDF, throttle)
├── matcher/               # Stage 1: keyword/phrase matching
├── judge/                 # Stage 2: LLM judge
│   ├── llm_judge.py       # Cerebras with key rotation
│   ├── discovery_agent.py # Stage 1.5: semantic discovery
│   └── prompt_templates/
├── contextual/            # Contextual AI integration
│   ├── contextual_client.py
│   └── metadata.py        # 20-field metadata builder
├── keys/                  # API key rotation
├── reddit/                # Reddit posting (daily only)
└── pipeline/
    ├── daily.py           # Daily RSS ingestion
    ├── backfill.py        # Historical backfill (citation sorting, top-N, auto-continuation)
    ├── updates.py         # Weekly updates
    ├── citations.py       # Citation refresh (OpenAlex)
    └── venue.py           # Top venue detection (auto-ingest bypass)

backfill/                  # Workflow input parser (parse_inputs.py)
streamlit_backfill/        # Backfill request UI (Streamlit Cloud)
```
