# Contextual arXiv Feed

A production-grade ingestion system that continuously ingests arXiv papers into a [Contextual AI](https://contextual.ai) Datastore using a multi-stage filtering pipeline.

## How It Works

Papers flow through a 3-stage filter before ingestion:

| Stage | Input | Method | Purpose |
|-------|-------|--------|---------|
| **Stage 1** | Title + RSS snippet | Keyword/phrase matching (stemmed) | Wide net — catches obvious matches |
| **Stage 1.5** | Title + snippet | Discovery Agent (LLM) | Semantic catch — finds papers keywords miss |
| **Stage 2** | Full abstract | Judge Agent (LLM) | Full topicality + quality evaluation |

A paper is accepted when `topicality_verdict == "accept" AND (quality_i >= 65 OR confidence_i < 80)`.

---

## 1. Daily Pipeline

Runs automatically via GitHub Actions at 06:00 UTC. Fetches new papers from arXiv RSS, filters through all 3 stages, ingests accepted papers to Contextual AI, and optionally posts top papers to Reddit.

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

Three modes for ingesting papers from the past. Backfill never posts to Reddit — silent ingestion only.

### Date range

Searches arXiv for papers updated within the given range (inclusive on both sides).

```bash
contextual-arxiv-feed backfill --start 2026-01-01 --end 2026-01-31
```

```bash
contextual-arxiv-feed backfill --start 2026-03-01 --end 2026-03-03 --dry-run
```

### Single date

```bash
contextual-arxiv-feed backfill-date --date 2026-03-10
```

```bash
contextual-arxiv-feed backfill-date --date 2026-03-10 --dry-run
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

Lightweight web UI for requesting backfills. Creates a GitHub Issue with a JSON payload — does NOT run ingestion. The [backfill.yml](.github/workflows/backfill.yml) workflow picks up the issue.

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

Features:
- Three modes: Single Date, Date Range, Identifiers (with validation preview)
- Creates issue with labels `backfill` + `streamlit-request`
- Monochrome theme (white/black, monospace)
- JSON payload in issue body, parseable by backfill workflow

#### Issue Payload Schema

```json
{
  "request_type": "date_range",
  "date": "",
  "start_date": "2026-03-01",
  "end_date": "2026-03-05",
  "identifiers": [],
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

LLM Judge (Cerebras primary, Gemini fallback):

```bash
export LLM_API_KEYS="your-cerebras-key"
export LLM_BASE_URL="https://api.cerebras.ai/v1"
export LLM_SECONDARY_API_KEYS="your-gemini-key"
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

3-tier fallback: Cerebras -> Gemini -> Local Qwen (if available).

### Reddit ([config/reddit.yaml](config/reddit.yaml))

Posts top daily papers to your subreddit. Daily pipeline only — backfill never posts.

---

## GitHub Actions Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| [daily.yml](.github/workflows/daily.yml) | Daily 06:00 UTC + manual | Full daily pipeline |
| [backfill.yml](.github/workflows/backfill.yml) | Manual only | Backfill from dispatch inputs or GitHub Issue |
| [weekly_updates.yml](.github/workflows/weekly_updates.yml) | Sunday 08:00 UTC | Version checks + DOI enrichment |
| [weekly_citations.yml](.github/workflows/weekly_citations.yml) | Saturday 10:00 UTC | Citation count refresh |
| [config_from_issue.yml](.github/workflows/config_from_issue.yml) | Issue with config labels | Validate & create PR |

### Backfill Workflow

Trigger manually with inputs or point to a GitHub Issue:

- **Direct inputs**: mode, date, start_date, end_date, identifiers, dry_run
- **Issue-based**: provide `issue_number` — reads JSON payload from issue body, requires `backfill` label

### Required Secrets

- `CONTEXTUAL_API_KEY`
- `CONTEXTUAL_DATASTORE_ID`

### Optional Secrets

- `LLM_API_KEYS`, `LLM_SECONDARY_API_KEYS` (judge)
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USERNAME`, `REDDIT_PASSWORD`
- `OPENALEX_API_KEYS`, `S2_API_KEYS` (citations)

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
│   ├── llm_judge.py       # 3-tier fallback: Cerebras → Gemini → local
│   ├── discovery_agent.py # Stage 1.5: semantic discovery
│   └── prompt_templates/
├── contextual/            # Contextual AI integration
│   ├── contextual_client.py
│   └── metadata.py        # 20-field metadata builder
├── keys/                  # API key rotation
├── reddit/                # Reddit posting (daily only)
└── pipeline/
    ├── daily.py           # Daily RSS ingestion
    ├── backfill.py        # Historical backfill (date range / identifiers)
    ├── updates.py         # Weekly updates
    └── citations.py       # Citation refresh

backfill/                  # Workflow input parser
streamlit_backfill/        # Backfill request UI
```
