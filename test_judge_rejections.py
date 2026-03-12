"""Quick test: does the judge actually reject any papers from today's RSS?

Parallelized: 4 threads, one per API key, each with independent rate limiting.
No Contextual AI ingestion.
"""

import logging
import os
import re
import json
import time
import threading
from pathlib import Path
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
)
logger = logging.getLogger(__name__)

from contextual_arxiv_feed.config import load_config
from contextual_arxiv_feed.arxiv import ArxivFeedParser, ArxivAPI
from contextual_arxiv_feed.arxiv.throttle import ArxivThrottle
from contextual_arxiv_feed.matcher import KeywordMatcher
from contextual_arxiv_feed.judge.judge import TEMPLATES_DIR
from contextual_arxiv_feed.judge.schema import JudgeOutput, parse_judge_output, truncate_rationale
from contextual_arxiv_feed.judge import JudgeResult

config_dir = Path(__file__).parent / "config"
config = load_config(config_dir)

throttle = ArxivThrottle(config.arxiv_throttle_seconds)
feed_parser = ArxivFeedParser(throttle)
api = ArxivAPI(throttle)
matcher = KeywordMatcher(config.topics)

keys_str = os.getenv("LLM_API_KEYS", "")
if keys_str:
    API_KEYS = [k.strip() for k in keys_str.split(",") if k.strip()]
else:
    single = os.getenv("LLM_API_KEY", "")
    API_KEYS = [single] if single else []

BASE_URL = os.getenv("LLM_BASE_URL", "https://api.cerebras.ai/v1")
MODEL_ID = os.getenv("LLM_MODEL_ID", "gpt-oss-120b")

logger.info(f"Keys: {len(API_KEYS)}, Model: {MODEL_ID}, Base: {BASE_URL}")

# ---------- Load prompt template ----------
template_path = TEMPLATES_DIR / "base_prompt.txt"
with open(template_path) as f:
    PROMPT_TEMPLATE = f.read()

thresholds = config.judge.get_thresholds()


def build_prompt(title: str, abstract: str) -> str:
    return PROMPT_TEMPLATE.format(
        title=title,
        abstract=abstract,
        strictness=config.judge.strictness,
        min_quality_i=thresholds.min_quality_i,
        prompt_version=config.judge.prompt_version,
        model_id=MODEL_ID,
        max_rationale_length=config.judge.max_rationale_length,
    )


def parse_response(response: str) -> JudgeOutput | None:
    """Parse LLM response to extract JSON."""
    if not response or not response.strip():
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = response
    json_match = re.search(r"```json\s*(.*?)\s*```", cleaned, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return None
    data = json.loads(json_str)
    # Cast numerics to int
    for field in ["prompt_version", "quality_i", "confidence_i"]:
        if field in data and isinstance(data[field], (int, float)):
            data[field] = int(round(data[field]))
    breakdown = data.get("quality_breakdown_i", {})
    if isinstance(breakdown, dict):
        for key in ["novelty_i", "relevance_i", "technical_depth_i"]:
            if key in breakdown and isinstance(breakdown[key], (int, float)):
                breakdown[key] = int(round(breakdown[key]))
    return parse_judge_output(data)


def worker_judge(api_key: str, key_idx: int, papers: list, results: list, lock: threading.Lock):
    """Worker thread: judges papers using a single dedicated API key.

    Each worker has its own httpx client and throttles to stay under 10 RPM.
    """
    client = httpx.Client(timeout=120.0)
    min_interval = 3.0  # 1 req per 3s = 20 RPM, under Cerebras 30 RPM limit
    last_call = 0.0
    key_suffix = api_key[-4:]

    for paper_idx, title, abstract, entry, topics in papers:
        prompt = build_prompt(title, abstract)

        # Throttle
        now = time.monotonic()
        elapsed = now - last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        # Retry loop for this single key
        max_retries = 10
        for attempt in range(max_retries):
            last_call = time.monotonic()
            try:
                resp = client.post(
                    f"{BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL_ID,
                        "max_tokens": 4096,
                        "temperature": 0.1,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Parse retry delay from response
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", e.response.text, re.IGNORECASE)
                    wait = float(match.group(1)) + 5 if match else 65
                    logger.info(f"[key{key_idx}...{key_suffix}] 429, waiting {wait:.0f}s (attempt {attempt+1})")
                    time.sleep(wait)
                    continue
                else:
                    logger.error(f"[key{key_idx}...{key_suffix}] HTTP {e.response.status_code}")
                    raw = ""
                    break
            except Exception as e:
                logger.error(f"[key{key_idx}...{key_suffix}] Error: {e}")
                raw = ""
                break
        else:
            logger.error(f"[key{key_idx}...{key_suffix}] Exhausted retries for {entry.arxiv_id}")
            with lock:
                results.append(("error", paper_idx, entry, None, topics, "exhausted retries"))
            continue

        # Parse
        try:
            output = parse_response(raw)
        except Exception as e:
            logger.error(f"Parse error: {e}")
            output = None

        if output is None:
            with lock:
                results.append(("error", paper_idx, entry, None, topics, "parse failed"))
            continue

        # Truncate rationale
        output_dict = output.to_dict()
        output_dict["rationale"] = truncate_rationale(
            output_dict["rationale"], config.judge.max_rationale_length
        )
        output = parse_judge_output(output_dict)

        quality_ok = output.quality_i >= 65
        low_confidence = output.confidence_i < 80

        if quality_ok or low_confidence:
            reason = "quality" if quality_ok else "low_confidence"
            logger.info(f"[{paper_idx}/{len(papers)*len(API_KEYS)}] ACCEPTED ({reason}): q={output.quality_i} c={output.confidence_i} | {entry.title[:60]}")
            with lock:
                results.append(("accepted", paper_idx, entry, output, topics, reason))
        else:
            logger.info(f"[{paper_idx}/{len(papers)*len(API_KEYS)}] REJECTED: q={output.quality_i} c={output.confidence_i} | {entry.title[:60]}")
            with lock:
                results.append(("rejected", paper_idx, entry, output, topics, ""))

    client.close()


# ---------- Fetch RSS ----------
categories = list({cat for t in config.topics.get_enabled_topics() for cat in t.arxiv_categories})
logger.info(f"Fetching RSS for {len(categories)} categories...")
entries = feed_parser.fetch_multiple_feeds(categories)
logger.info(f"Total RSS entries: {len(entries)}")

# Stage 1: keyword match
passed = []
for entry in entries:
    match = matcher.match(entry.title, entry.abstract_snippet)
    if match.passes_stage1:
        passed.append((entry, match.matched_topics))

logger.info(f"Stage 1 passed: {len(passed)} / {len(entries)}")

# ---------- Fetch full abstracts (batch mode, ~50 per request) ----------
BATCH_SIZE = 50
papers_with_abstracts = []

# Build ID-to-entry mapping
id_to_info = {}
for i, (entry, topics) in enumerate(passed):
    id_to_info[entry.id_with_version] = (i + 1, entry, topics)

# Batch fetch
all_ids = [entry.id_with_version for entry, _ in passed]
fetched_metadata = {}  # arxiv_id -> ArxivMetadata

for batch_start in range(0, len(all_ids), BATCH_SIZE):
    batch_ids = all_ids[batch_start : batch_start + BATCH_SIZE]
    batch_num = batch_start // BATCH_SIZE + 1
    total_batches = (len(all_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Fetching batch {batch_num}/{total_batches} ({len(batch_ids)} papers)...")
    try:
        results = api.fetch_by_ids(batch_ids)
        for meta in results:
            fetched_metadata[meta.arxiv_id] = meta
        logger.info(f"  Got {len(results)} results")
    except Exception as e:
        logger.warning(f"Batch {batch_num} failed ({e}), using RSS snippets for this batch")

# Build final list
for arxiv_id in all_ids:
    idx, entry, topics = id_to_info[arxiv_id]
    meta = fetched_metadata.get(entry.arxiv_id)
    if meta:
        title = meta.title
        abstract = meta.abstract
    else:
        title = entry.title
        abstract = entry.abstract_snippet or "(no abstract)"
    papers_with_abstracts.append((idx, title, abstract, entry, topics))

full_abstract_count = sum(1 for aid in all_ids if id_to_info[aid][1].arxiv_id in fetched_metadata)
logger.info(f"Metadata fetched: {full_abstract_count}/{len(passed)} have full abstracts")

# ---------- Distribute papers across workers ----------
num_workers = min(len(API_KEYS), 4)
worker_papers = [[] for _ in range(num_workers)]
for i, paper in enumerate(papers_with_abstracts):
    worker_papers[i % num_workers].append(paper)

logger.info(f"Distributing {len(papers_with_abstracts)} papers across {num_workers} workers")
for i, wp in enumerate(worker_papers):
    logger.info(f"  Worker {i}: {len(wp)} papers")

# ---------- Run parallel judge ----------
results = []
lock = threading.Lock()
threads = []

for i in range(num_workers):
    t = threading.Thread(
        target=worker_judge,
        args=(API_KEYS[i], i, worker_papers[i], results, lock),
        name=f"judge-{i}",
    )
    threads.append(t)
    t.start()
    time.sleep(0.5)  # stagger start slightly

for t in threads:
    t.join()

# ---------- Summary ----------
accepted = [(e, o, t) for status, _, e, o, t, _ in results if status == "accepted"]
rejected = [(e, o, t) for status, _, e, o, t, _ in results if status == "rejected"]
errors = [(e, r) for status, _, e, _, _, r in results if status == "error"]

print("\n" + "=" * 80)
print(f"JUDGE TEST RESULTS (parallel, {num_workers} workers)")
print(f"=" * 80)
print(f"  RSS entries:     {len(entries)}")
print(f"  Stage 1 passed:  {len(passed)}")
print(f"  Judge accepted:  {len(accepted)}")
print(f"  Judge REJECTED:  {len(rejected)}")
print(f"  Judge errors:    {len(errors)}")
print()

if rejected:
    print("REJECTED PAPERS:")
    print("-" * 60)
    for entry, out, topics in rejected:
        print(f"  {entry.arxiv_id}: {entry.title[:70]}")
        print(f"    quality={out.quality_i}, confidence={out.confidence_i}, topics={topics}")
        print(f"    verdict={out.quality_verdict}")
        print(f"    rationale: {out.rationale[:120]}")
        if out.quality_breakdown_i:
            b = out.quality_breakdown_i
            print(f"    breakdown: novelty={b.novelty_i}, relevance={b.relevance_i}, depth={b.technical_depth_i}")
        print()
else:
    print("NO PAPERS WERE REJECTED — judge accepted everything.")

if accepted:
    print(f"\nACCEPTED PAPERS ({len(accepted)}):")
    print("-" * 60)
    for entry, out, topics in accepted:
        print(f"  {entry.arxiv_id}: q={out.quality_i} c={out.confidence_i} | {entry.title[:60]}")

feed_parser.close()
api.close()
