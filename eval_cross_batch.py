"""Cross-batch evaluation: Cerebras scoring + Opus oracle comparison.

Fetches papers from arXiv, scores with Cerebras gpt-oss-120b,
outputs structured data for Opus evaluation across multiple batches.

Usage:
    # Score a batch of papers with Cerebras
    python eval_cross_batch.py score --date 2026-03-10 --limit 40

    # After Opus evaluates, analyze configs across all batches
    python eval_cross_batch.py analyze
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

EVAL_DIR = Path(__file__).parent / "eval_results" / "cross_batch"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Load env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def get_keys():
    keys_str = os.getenv("LLM_API_KEYS", "")
    if keys_str:
        return [k.strip() for k in keys_str.split(",") if k.strip()]
    single = os.getenv("LLM_API_KEY", "")
    return [single] if single else []


def fetch_arxiv_papers(date_str: str, categories: list[str], max_results: int = 40, offset: int = 0):
    """Fetch papers from arXiv API.

    Args:
        date_str: Not used for filtering (arXiv API doesn't support date filter well).
                  Used for metadata only.
        categories: arXiv categories to query.
        max_results: Number of papers to return.
        offset: Start offset for pagination (use to get different batches).
    """
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    query = f"({cat_query})"

    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": offset,
        "max_results": max_results * 3,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    print(f"Fetching papers from arXiv (categories: {categories})...")
    resp = httpx.get(url, params=params, timeout=30)
    resp.raise_for_status()

    import xml.etree.ElementTree as ET
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        arxiv_id_url = entry.find("atom:id", ns).text
        arxiv_id = arxiv_id_url.split("/abs/")[-1]
        published = entry.find("atom:published", ns).text[:10]

        cats = []
        for cat in entry.findall("atom:category", ns):
            cats.append(cat.get("term"))
        primary = entry.find("arxiv:primary_category", ns)
        primary_cat = primary.get("term") if primary is not None else (cats[0] if cats else "")

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "published": published,
            "categories": cats,
            "primary_category": primary_cat,
        })

    # Deduplicate and limit
    seen = set()
    unique = []
    for p in papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            unique.append(p)
    return unique[:max_results]


def score_with_cerebras(papers: list[dict], prompt_template: str, config: dict):
    """Score papers with Cerebras gpt-oss-120b."""
    keys = get_keys()
    if not keys:
        print("ERROR: No LLM_API_KEYS set", file=sys.stderr)
        sys.exit(1)

    base_url = os.getenv("LLM_BASE_URL", "https://api.cerebras.ai/v1")
    model_id = config.get("model_id", "gpt-oss-120b")
    key_idx = 0
    results = []
    client = httpx.Client(timeout=120.0)

    for i, paper in enumerate(papers):
        prompt = prompt_template.format(
            title=paper["title"],
            abstract=paper["abstract"],
            strictness="medium",
            min_quality_i=65,
            prompt_version=5,
            model_id=model_id,
            max_rationale_length=300,
        )

        # Try with key rotation
        for attempt in range(len(keys) * 2):
            api_key = keys[key_idx % len(keys)]
            try:
                time.sleep(2.5)  # Rate limit: ~24 RPM across 4 keys
                resp = client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "max_tokens": 4096,
                        "temperature": 0.1,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]

                # Parse JSON from response
                cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                json_match = re.search(r"```json\s*(.*?)\s*```", cleaned, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                    json_str = json_match.group(0) if json_match else None

                if json_str:
                    data = json.loads(json_str)
                    # Cast to int
                    for f in ["quality_i", "confidence_i", "prompt_version"]:
                        if f in data:
                            data[f] = int(round(data[f]))
                    bd = data.get("quality_breakdown_i", {})
                    for f in ["novelty_i", "relevance_i", "technical_depth_i"]:
                        if f in bd:
                            bd[f] = int(round(bd[f]))

                    paper["cerebras_quality"] = data["quality_i"]
                    paper["cerebras_confidence"] = data["confidence_i"]
                    paper["cerebras_verdict"] = data["quality_verdict"]
                    paper["cerebras_novelty"] = bd.get("novelty_i", 0)
                    paper["cerebras_relevance"] = bd.get("relevance_i", 0)
                    paper["cerebras_technical_depth"] = bd.get("technical_depth_i", 0)
                    paper["cerebras_rationale"] = data.get("rationale", "")
                    results.append(paper)
                    print(f"  [{i+1}/{len(papers)}] q={data['quality_i']} c={data['confidence_i']} | {paper['title'][:60]}")
                    break
                else:
                    print(f"  [{i+1}] No JSON in response, retrying...")
                    key_idx += 1

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    retry_match = re.search(r"retry in (\d+(?:\.\d+)?)s", e.response.text, re.IGNORECASE)
                    delay = float(retry_match.group(1)) + 5 if retry_match else 65
                    print(f"  429 on key #{key_idx % len(keys)}, waiting {delay:.0f}s...")
                    key_idx += 1
                    time.sleep(delay)
                else:
                    print(f"  HTTP error: {e.response.status_code}", file=sys.stderr)
                    key_idx += 1
        else:
            print(f"  [{i+1}] FAILED after all attempts: {paper['title'][:50]}")

    client.close()
    return results


def score_batch(date_str: str, limit: int = 40, offset: int = 0):
    """Fetch and score a batch of papers."""
    categories = ["cs.CL", "cs.AI", "cs.LG", "cs.IR"]

    papers = fetch_arxiv_papers(date_str, categories, limit, offset)
    if not papers:
        print("No papers found")
        return

    print(f"Found {len(papers)} papers, scoring with Cerebras...")

    # Load prompt template
    template_path = Path(__file__).parent / "src" / "contextual_arxiv_feed" / "judge" / "prompt_templates" / "base_prompt.txt"
    prompt_template = template_path.read_text()

    scored = score_with_cerebras(papers, prompt_template, {"model_id": "gpt-oss-120b"})

    # Save batch
    batch_id = f"batch_{date_str}_{datetime.now().strftime('%H%M%S')}"
    batch_path = EVAL_DIR / f"{batch_id}.json"

    batch_data = {
        "batch_id": batch_id,
        "date": date_str,
        "scored_at": datetime.now().isoformat(),
        "model": "gpt-oss-120b",
        "paper_count": len(scored),
        "papers": scored,
    }

    with open(batch_path, "w") as f:
        json.dump(batch_data, f, indent=2)

    print(f"\nBatch saved: {batch_path}")
    print(f"Papers scored: {len(scored)}")

    # Print for Opus evaluation
    print("\n" + "=" * 80)
    print("PAPERS FOR OPUS EVALUATION")
    print("For each paper, decide: is this relevant to our 6 topics?")
    print("Topics: context-engineering, rag-retrieval, llm-inference,")
    print("        agents-tools, fine-tuning, generative-ai-foundations")
    print("=" * 80)
    for i, p in enumerate(scored):
        print(f"\n--- Paper {i+1} ---")
        print(f"Title: {p['title']}")
        print(f"Abstract: {p['abstract'][:500]}")
        print(f"Categories: {p['primary_category']}")
        print(f"Cerebras: q={p['cerebras_quality']} c={p['cerebras_confidence']}")

    return batch_path


def analyze_all_batches():
    """Analyze configs across all batches that have Opus evaluations."""
    batch_files = sorted(EVAL_DIR.glob("batch_*.json"))
    if not batch_files:
        print("No batch files found in eval_results/cross_batch/")
        return

    all_batches = []
    for bf in batch_files:
        with open(bf) as f:
            data = json.load(f)
        # Check if Opus evaluations exist
        if not any("opus_relevant" in p for p in data["papers"]):
            print(f"  SKIP {bf.name}: no Opus evaluations yet")
            continue
        all_batches.append(data)
        print(f"  LOADED {bf.name}: {len(data['papers'])} papers")

    if not all_batches:
        print("No batches with Opus evaluations found.")
        return

    # Define configs to test
    configs = {
        "q>=60": lambda p: p["cerebras_quality"] >= 60,
        "q>=62": lambda p: p["cerebras_quality"] >= 62,
        "q>=63": lambda p: p["cerebras_quality"] >= 63,
        "q>=65": lambda p: p["cerebras_quality"] >= 65,
        "q>=67": lambda p: p["cerebras_quality"] >= 67,
        "q>=70": lambda p: p["cerebras_quality"] >= 70,
        "q>=65 OR c<80": lambda p: p["cerebras_quality"] >= 65 or p["cerebras_confidence"] < 80,
        "q>=65 OR (q>=62 AND c<=78)": lambda p: p["cerebras_quality"] >= 65 or (p["cerebras_quality"] >= 62 and p["cerebras_confidence"] <= 78),
        "q>=65 AND NOT (q<70 AND c>=85)": lambda p: p["cerebras_quality"] >= 65 and not (p["cerebras_quality"] < 70 and p["cerebras_confidence"] >= 85),
        "(q>=65 OR (q>=62 AND c<=78)) AND NOT (q<70 AND c>=85)": lambda p: (p["cerebras_quality"] >= 65 or (p["cerebras_quality"] >= 62 and p["cerebras_confidence"] <= 78)) and not (p["cerebras_quality"] < 70 and p["cerebras_confidence"] >= 85),
        "q>=62 AND NOT (q<70 AND c>=85)": lambda p: p["cerebras_quality"] >= 62 and not (p["cerebras_quality"] < 70 and p["cerebras_confidence"] >= 85),
    }

    # Per-batch metrics
    print("\n" + "=" * 100)
    print("CROSS-BATCH CONFIG COMPARISON")
    print("=" * 100)

    config_all_f1s = {name: [] for name in configs}
    config_all_precisions = {name: [] for name in configs}
    config_all_recalls = {name: [] for name in configs}

    for batch in all_batches:
        papers = [p for p in batch["papers"] if "opus_relevant" in p]
        print(f"\n--- {batch['batch_id']} ({len(papers)} papers) ---")
        print(f"  {'Config':<55s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'FP':>4s} {'FN':>4s}")

        for config_name, accept_fn in configs.items():
            accepted = [p for p in papers if accept_fn(p)]
            rejected = [p for p in papers if not accept_fn(p)]

            tp = sum(1 for p in accepted if p["opus_relevant"])
            fp = sum(1 for p in accepted if not p["opus_relevant"])
            fn = sum(1 for p in rejected if p["opus_relevant"])

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            config_all_f1s[config_name].append(f1)
            config_all_precisions[config_name].append(precision)
            config_all_recalls[config_name].append(recall)

            print(f"  {config_name:<55s} {precision:>5.1%} {recall:>5.1%} {f1:>5.1%} {fp:>4d} {fn:>4d}")

    # Aggregate: mean + std across batches
    print("\n" + "=" * 100)
    print("AGGREGATE ACROSS ALL BATCHES (mean ± std)")
    print("=" * 100)
    print(f"  {'Config':<55s} {'F1 mean':>8s} {'F1 std':>7s} {'Prec':>6s} {'Recall':>6s} {'Stable?':>8s}")
    print("-" * 100)

    import statistics
    best_score = 0
    best_config = ""

    for config_name in configs:
        f1s = config_all_f1s[config_name]
        precs = config_all_precisions[config_name]
        recs = config_all_recalls[config_name]

        f1_mean = statistics.mean(f1s)
        f1_std = statistics.stdev(f1s) if len(f1s) > 1 else 0
        prec_mean = statistics.mean(precs)
        rec_mean = statistics.mean(recs)

        # Stability score: high F1 mean with low variance
        # Penalize configs with high variance
        stability = f1_mean - f1_std
        stable = "YES" if f1_std < 0.05 else "no"

        print(f"  {config_name:<55s} {f1_mean:>7.1%} {f1_std:>6.1%} {prec_mean:>5.1%} {rec_mean:>5.1%} {stable:>8s}")

        if stability > best_score:
            best_score = stability
            best_config = config_name

    print(f"\n  BEST (highest F1 - std): {best_config}")

    # Save analysis
    analysis_path = EVAL_DIR / f"cross_batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    analysis = {
        "batches": [b["batch_id"] for b in all_batches],
        "batch_count": len(all_batches),
        "configs": {
            name: {
                "f1_per_batch": config_all_f1s[name],
                "precision_per_batch": config_all_precisions[name],
                "recall_per_batch": config_all_recalls[name],
                "f1_mean": statistics.mean(config_all_f1s[name]),
                "f1_std": statistics.stdev(config_all_f1s[name]) if len(config_all_f1s[name]) > 1 else 0,
            }
            for name in configs
        },
        "best_config": best_config,
    }
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved: {analysis_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-batch Cerebras vs Opus evaluation")
    sub = parser.add_subparsers(dest="command")

    score_cmd = sub.add_parser("score", help="Score a batch of papers with Cerebras")
    score_cmd.add_argument("--date", required=True, help="Date label for this batch (YYYY-MM-DD)")
    score_cmd.add_argument("--limit", type=int, default=40, help="Max papers to score")
    score_cmd.add_argument("--offset", type=int, default=0, help="Start offset for pagination (0, 35, 70, etc.)")

    sub.add_parser("analyze", help="Analyze all batches with Opus evaluations")

    args = parser.parse_args()
    if args.command == "score":
        score_batch(args.date, args.limit, args.offset)
    elif args.command == "analyze":
        analyze_all_batches()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
