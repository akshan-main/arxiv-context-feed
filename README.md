# Contextual arXiv Feed

A production-grade ingestion system that continuously ingests arXiv papers into a Contextual AI Datastore using a multi-stage filtering pipeline powered by a local LLM.

## Overview

This system automates the discovery and ingestion of relevant arXiv papers using a split architecture:

- **Oracle Cloud**: Runs LLM-heavy analysis (RSS fetch, keyword matching, Discovery Agent, Judge Agent, Reddit posting)
- **GitHub Actions**: Handles network I/O (PDF download + Contextual AI ingestion), triggered by GitHub Issues
- **Communication**: Via GitHub Issues — each daily run creates an issue as an audit trail

### Multi-Stage Filtering

| Stage | Input | Method | Purpose |
|-------|-------|--------|---------|
| **Stage 1** | Title + RSS snippet | Keyword/phrase matching (stemmed) | Wide net — catches obvious matches |
| **Stage 1.5** | Title + snippet | Discovery Agent (local LLM) | Semantic catch — finds papers keywords miss |
| **Stage 2** | Full abstract | Judge Agent (local LLM) | Full topicality + quality evaluation |

**Acceptance Criteria**: `topicality_verdict == "accept" AND (quality_i >= 65 OR confidence_i < 80)`

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/arxiv-context-feed.git
cd arxiv-context-feed

# Install base dependencies
pip install -e .

# Install with local LLM support (for Oracle Cloud)
pip install -e ".[local-llm]"

# Install with Reddit posting support
pip install -e ".[reddit]"
```

### Environment Variables

```bash
# Required for ingestion (GitHub Actions)
export CONTEXTUAL_API_KEY="your-contextual-api-key"
export CONTEXTUAL_DATASTORE_ID="your-datastore-id"

# Local LLM (Oracle Cloud — defaults work out of the box)
export LLM_BASE_URL="http://127.0.0.1:8080/v1"  # Default
export LLM_API_KEY="not-needed"                   # Default for local

# Optional
export ARXIV_THROTTLE_SECONDS="3"
export OPENALEX_API_KEYS="key1,key2,key3"  # Comma-separated for rotation
export DRY_RUN="true"  # Skip actual uploads

# Reddit posting (optional)
export REDDIT_CLIENT_ID="your-client-id"
export REDDIT_CLIENT_SECRET="your-client-secret"
export REDDIT_USERNAME="your-bot-username"
export REDDIT_PASSWORD="your-bot-password"
```

### Running the Pipelines

```bash
# Run analysis only (Oracle Cloud — no PDF download/ingestion)
contextual-arxiv-feed run-analyze

# Run daily RSS feed ingestion (full pipeline)
contextual-arxiv-feed run-daily

# Run weekly updates (new versions + DOI enrichment)
contextual-arxiv-feed run-updates --lookback-days 7

# Refresh citations for papers with DOI
contextual-arxiv-feed refresh-citations

# Backfill papers from a date range
contextual-arxiv-feed backfill --start 2024-01-01 --end 2024-01-31

# Dry run (no uploads)
contextual-arxiv-feed dry-run --mode daily
```

## Split Architecture

```
Oracle Cloud (daily cron 06:00 UTC)
  ├── Fetch RSS feeds
  ├── Stage 1: Keyword matcher
  ├── Stage 1.5: Discovery Agent (semantic catch for keyword misses)
  ├── Dedup + idempotency check
  ├── Stage 2: Judge Agent (full evaluation, 3 scores)
  ├── Post top papers to Reddit (own subreddit)
  └── Creates GitHub Issue with accepted papers (JSON payload)
         │
         ↓  (issue label triggers workflow)
GitHub Actions
  ├── Parse issue body → list of accepted papers
  ├── Download PDFs from arXiv
  ├── Ingest to Contextual AI datastore
  ├── Comment on issue with ingestion results
  └── ~15 min/run = 450 min/month (within 2,000 free limit)
```

## Configuration

### Topics (`config/topics.yaml`)

6 topic groups covering ~40+ concepts:

| Topic | Key | Covers |
|-------|-----|--------|
| Context Engineering | `context-engineering` | Context windows, poisoning, compression, attention |
| RAG & Retrieval | `rag-retrieval` | RAG, embeddings, vector DB, document parsing, re-ranking |
| LLM Inference | `llm-inference` | Inference optimization, reasoning, chain-of-thought |
| Agents & Tools | `agents-tools` | LLM agents, tool use, multi-agent, scaffolding |
| Fine-tuning | `fine-tuning` | Fine-tuning, alignment, prompt engineering, PEFT |
| Gen AI Foundations | `generative-ai-foundations` | Transformers, NLP, chatbots, language generation |

Stage 1 keywords cast a WIDE net. The LLM judge provides semantic understanding.

### Judge (`config/judge.yaml`)

```yaml
provider: local
model_id: qwen2.5-14b-instruct-q4_k_m
strictness: medium
prompt_version: 3
max_rationale_length: 300
```

### Reddit (`config/reddit.yaml`)

```yaml
enabled: false
subreddits: []  # Your own subreddit
min_quality_i: 75
max_posts_per_run: 5
flair_overrides: {}  # Auto-derived from topics.yaml
```

### API Key Rotation

For teams with multiple API keys, use comma-separated values:

```bash
export OPENALEX_API_KEYS="key1,key2,key3,key4,key5"
```

Keys are rotated round-robin with automatic cooldown on rate limits.

## Oracle Cloud Deployment

```bash
# One-time setup on Oracle Cloud Always Free (4 ARM cores, 24GB RAM)
bash deploy/setup_oracle.sh

# Health check
bash deploy/healthcheck.sh
```

The local LLM (Qwen2.5-14B Q4) runs as a systemd service via llama-cpp-python.

## Chatbot

A separate open-source chatbot lives in `chatbot/`. It has its own datastore (FAISS + sentence-transformers), cross-encoder reranking, pluggable generator, and Whisper voice input. Deploy to HuggingFace Spaces.

## Metadata Architecture

### Document Naming

- **PDF**: `arxiv:{arxiv_id}v{version}` (e.g., `arxiv:2401.12345v1`)
- **Manifest**: `arxiv:{arxiv_id}v{version}:manifest`

### custom_metadata

All numeric fields are integers (INT-ONLY enforcement).

```python
{
    "source": "arxiv",
    "arxiv_id": "2401.12345",
    "arxiv_version": 1,
    "title": "Paper Title",
    "topics": "context-engineering|rag-retrieval",
    "quality_i": 75,
    "topic_confidence_i": 85,
    "judge_model_id": "qwen2.5-14b-instruct-q4_k_m",
    "judge_prompt_version": 3,
    # ... more fields
}
```

## GitHub Actions Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `daily.yml` | Issue labeled `daily-ingestion` | PDF download + ingestion |
| `weekly_updates.yml` | Sunday 08:00 UTC | Version checks + DOI enrichment |
| `weekly_citations.yml` | Saturday 10:00 UTC | Citation count refresh |
| `config_from_issue.yml` | Issue with config labels | Validate & create PR |
| `deploy.yml` | Push to main | SSH deploy to Oracle Cloud |

### Required Secrets

- `CONTEXTUAL_API_KEY`
- `CONTEXTUAL_DATASTORE_ID`

### Optional Secrets

- `OPENALEX_API_KEYS`
- `ORACLE_SSH_KEY` (for deploy workflow)
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USERNAME`, `REDDIT_PASSWORD`

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Architecture

```
src/contextual_arxiv_feed/
├── config.py              # Pydantic config models
├── cli.py                 # Click CLI
├── arxiv/                 # arXiv integration (feeds, API, PDF, throttle)
├── matcher/               # Stage 1: keyword/phrase matching
├── judge/                 # Stage 2: LLM judge
│   ├── llm_judge.py       # LLM judge (3-tier fallback: Cerebras → Gemini → local Qwen)
│   ├── discovery_agent.py # Stage 1.5: semantic discovery
│   ├── schema.py          # Judge output validation
│   └── prompt_templates/
├── llm/                   # Local LLM server management
├── keys/                  # API key rotation (round-robin + cooldown)
├── reddit/                # Reddit posting agent
├── contextual/            # Contextual AI integration
└── pipeline/              # Pipeline orchestration
    ├── daily.py           # Daily RSS ingestion
    ├── updates.py         # Weekly updates
    ├── citations.py       # Citation refresh (with key rotation)
    ├── backfill.py        # Historical backfill
    └── apply_config_change.py

chatbot/                   # Separate: open-source Chainlit chatbot
deploy/                    # Oracle Cloud deployment configs
```s
