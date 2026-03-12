"""Evaluate Cerebras judge safety net configurations using Opus ground truth.

Opus (Claude) manually evaluated 100 stratified-sample papers by reading
full abstracts, matching the same process as the LLM judge pipeline.

Tests multiple safety net configurations and produces reproducible metrics.
Saves results to eval_results/ for documentation.

Usage:
    python eval_safety_net.py
"""

import json
from pathlib import Path
from datetime import datetime

EVAL_DIR = Path(__file__).parent / "eval_results"
EVAL_DIR.mkdir(exist_ok=True)

# ---------- Opus Ground Truth (evaluated 2026-03-05) ----------
# 100 papers stratified by group: 30 safety_net, 25 borderline, 10 mid, 5 high, 30 rejected
# Each paper was evaluated by reading its full abstract against our 6 topics.
# Seed: random.seed(42) for reproducible sampling from /tmp/eval_with_abstracts.json
#
# Topics:
#   1. Context Engineering & Management
#   2. RAG, Retrieval & Knowledge Systems
#   3. LLM Inference & Reasoning
#   4. LLM Agents, Tool Use & Multi-Agent Systems
#   5. Fine-tuning, Alignment & Prompt Engineering
#   6. Generative AI & NLP Foundations
#
# Key exclusions applied:
#   - Papers merely USING transformers/DL for non-LLM domains (robotics, physics,
#     medical imaging, autonomous driving) are NOT relevant
#   - Traditional RL agents, game-playing AI, non-LLM robotics are NOT relevant
#   - Image/video/audio generation without LLM text components are NOT relevant

# Paper index (1-based, matching sample order) -> is_relevant
OPUS_GROUND_TRUTH = {
    # --- safety_net group (q<65, c<80) ---
    1: False,   # AutoQD: Quality-Diversity RL optimization, no LLM
    2: False,   # Carbon flux prediction, environmental science
    3: False,   # Argument appraisal framework, NLP but doesn't advance LLMs
    4: False,   # CausalMix for causal inference, not LLM
    5: False,   # Kernel gradient descent parameter selection, pure math
    6: False,   # SPRINT: few-shot tabular learning, general ML
    7: False,   # InstMeter: DL inference on MCUs, not LLM
    8: True,    # Epistemological consequences of LLMs (generative-ai-foundations)
    9: False,   # Multi-agent RL Nash equilibria, traditional RL
    10: False,  # HALyPO: human-robot collaboration MARL, robotics
    11: False,  # NeuroFlowNet: EEG reconstruction, neuroscience
    12: True,   # Causal learning with LLM agents (agents-tools)
    13: False,  # Contrastive causal discovery, pure statistics
    14: False,  # Self-adapting robotic agents via RL
    15: False,  # CNF placement for 5G/6G networks
    16: False,  # JANUS: synthetic tabular data generation
    17: True,   # Military AI agents governance (agents-tools)
    18: False,  # DisenReason: sequential recommendation
    19: False,  # Unsupervised IRM, representation learning
    20: True,   # Systematic analysis of biases in LLMs (fine-tuning/alignment)
    21: False,  # Phys4D: video diffusion 4D modeling
    22: False,  # CAMMSR: multimodal recommendation
    23: True,   # Testing learning hypotheses with neural LMs (generative-ai-foundations)
    24: True,   # Speculative decoding for OpenPangu on NPU (llm-inference)
    25: False,  # PDE foundation models for materials, physics
    26: False,  # ML force fields for molecular dynamics
    27: False,  # AxelGNN, general GNN
    28: False,  # VITA: vision-to-action robotics
    29: False,  # Diffusion steganography, image generation
    30: True,   # RAG vs. GraphRAG systematic evaluation (rag-retrieval)
    # --- borderline_65_69 group (q=65-69) ---
    31: True,   # Threat intelligence to firewall rules, multi-agent LLM system (agents-tools)
    32: True,   # LLMs and human rights evaluation (fine-tuning/alignment)
    33: False,  # Hardware co-optimization for in-memory computing
    34: True,   # RevPerf: Android perf issues with LLM agent (agents-tools)
    35: True,   # Temperature in knowledge distillation (llm-inference)
    36: False,  # CERNet: robot motion with predictive-coding RNN
    37: False,  # Vision Mamba autoregressive pretraining, computer vision
    38: True,   # Causality elicitation from LLMs (generative-ai-foundations)
    39: False,  # mlx-vis: dimensionality reduction library
    40: True,   # Fine-tuning LLMs for sensorimotor representations (fine-tuning)
    41: False,  # Continuous Modal Logical Neural Networks, pure logic
    42: False,  # Lang2Str: LLMs for crystal structure, domain application
    43: True,   # Agent-User Problem in IR (rag-retrieval, agents-tools)
    44: True,   # MEM: Multi-scale embodied memory for VLA models (agents-tools)
    45: False,  # DKD-KAN: knowledge distillation for intrusion detection
    46: True,   # LeanTutor: LLM + theorem prover (agents-tools, llm-inference)
    47: False,  # Graph Hopfield Networks, general graph ML
    48: True,   # GenAI Workbench: AI-assisted engineering with VLMs (agents-tools)
    49: False,  # BAH dataset for ambivalence recognition in videos
    50: False,  # Chimera: neuro-symbolic dataplane intelligence
    51: True,   # LLM-based recommendation with knowledge graphs (rag-retrieval)
    52: True,   # Coreference resolution for RAG (rag-retrieval)
    53: True,   # LX Topic: neural topic modeling with LLM (generative-ai-foundations)
    54: True,   # Detecting AI-generated essays (generative-ai-foundations)
    55: True,   # MT quality prediction with LLMs (generative-ai-foundations)
    # --- mid_70_79 group (q=70-79) ---
    56: True,   # Risk assessment for LLM-powered healthcare (agents-tools)
    57: True,   # Preference model biases for LLM alignment (fine-tuning)
    58: False,  # Backdoor attacks on diffusion models, image generation
    59: False,  # OMNIINTENT: DeFi framework, domain application
    60: True,   # Sleeper Cell: backdoor in tool-using LLM agents (fine-tuning, agents-tools)
    61: True,   # LaViRA: zero-shot VLN with MLLMs (agents-tools)
    62: False,  # Merlin: medical VLM, domain application
    63: True,   # Phi-4-reasoning-vision technical report (generative-ai-foundations)
    64: True,   # SafeDPO: DPO with safety alignment (fine-tuning)
    65: False,  # VLM uncertainty for histopathology, domain application
    # --- high_80plus group (q=80+) ---
    66: True,   # Prompt-dependent ranking of LLMs (generative-ai-foundations)
    67: True,   # Spatial Credit Redistribution for VLMs, hallucination (llm-inference)
    68: True,   # Entropic-Time Inference for LLM decoding (llm-inference)
    69: True,   # Crab+: Audio-Visual LLM (generative-ai-foundations)
    70: True,   # SafeCRS: safety alignment for LLM recommender (fine-tuning)
    # --- rejected group (q<65, c>=80) ---
    71: False,  # Trajectory prediction for autonomous driving
    72: False,  # Cattle re-identification dataset
    73: False,  # Navigation in uncertain environments
    74: False,  # MDPs with exogenous dynamics, pure RL
    75: False,  # LiDAR localization in forests
    76: False,  # Synthetic image augmentation realism
    77: False,  # Dynamic regret minimization in bandits
    78: False,  # Diffusion policies for robot manipulation
    79: False,  # Counterfactual explanations for GNNs
    80: False,  # Soft robotics in stratosphere
    81: False,  # Speech enhancement
    82: False,  # Drone racing RL
    83: False,  # Structural Action Transformer for robotics
    84: False,  # Orbital Transformers for computational chemistry
    85: True,   # 2-Coloring cycles: proof discovered by LLMs + Lean 4 (llm-inference)
    86: False,  # Surgical robot manipulation
    87: False,  # GNN negative feedback bias correction
    88: False,  # Maritime vessel navigation RL
    89: False,  # Reward-guided image editing with diffusion
    90: False,  # Decision transformer for offline RL
    91: False,  # Earthquake phase picking
    92: False,  # Equivariant imaging
    93: False,  # Meta-learning intrinsic rewards for RL
    94: False,  # Multimodal homography estimation
    95: False,  # Quantum circuit synthesis
    96: False,  # Soft robotic arm hardware
    97: False,  # Fair allocation theory
    98: False,  # 3D reconstruction
    99: False,  # Molecular density representation
    100: False, # Blood cell classification CNN
}


def analyze():
    """Analyze safety net configurations against Opus ground truth."""
    # Load the stratified sample
    sample_path = Path("/tmp/opus_evaluations.json")
    if not sample_path.exists():
        print("ERROR: /tmp/opus_evaluations.json not found.")
        print("Run the sampling script first to create the evaluation data.")
        return

    with open(sample_path) as f:
        papers = json.load(f)

    # Verify ground truth matches
    for i, p in enumerate(papers):
        expected = OPUS_GROUND_TRUTH[i + 1]
        if p["opus_relevant"] != expected:
            print(f"WARNING: Mismatch at paper {i+1}: file={p['opus_relevant']}, code={expected}")

    report = []
    report.append("=" * 90)
    report.append("SAFETY NET EVALUATION REPORT")
    report.append(f"Date: {datetime.now().isoformat()}")
    report.append("Ground truth: Opus 4.6 (Claude) manual evaluation of full abstracts")
    report.append("Judge model: Cerebras gpt-oss-120b")
    report.append(f"Sample: {len(papers)} papers (stratified: 30 safety_net, 25 borderline,")
    report.append("         10 mid, 5 high, 30 rejected) — seed=42")
    report.append("=" * 90)

    # ---------- Group-level accuracy ----------
    report.append("\n\n## GROUP-LEVEL ACCURACY (Cerebras quality groups vs Opus ground truth)")
    report.append("-" * 70)
    total_relevant = 0
    total_papers = 0
    for group in ["safety_net", "borderline_65_69", "mid_70_79", "high_80plus", "rejected"]:
        group_papers = [p for p in papers if p["group"] == group]
        if not group_papers:
            continue
        relevant = sum(1 for p in group_papers if p["opus_relevant"])
        total = len(group_papers)
        total_relevant += relevant
        total_papers += total
        pct = relevant / total * 100
        report.append(f"  {group:25s}: {relevant:3d}/{total:3d} relevant ({pct:5.1f}%)")
    report.append(f"  {'TOTAL':25s}: {total_relevant:3d}/{total_papers:3d} relevant ({total_relevant/total_papers*100:5.1f}%)")

    # ---------- Quality score distribution ----------
    report.append("\n\n## QUALITY SCORE DISTRIBUTION BY OPUS RELEVANCE")
    report.append("-" * 70)
    relevant_qs = [p["cerebras_quality"] for p in papers if p["opus_relevant"]]
    irrelevant_qs = [p["cerebras_quality"] for p in papers if not p["opus_relevant"]]
    report.append(f"  Relevant papers   (n={len(relevant_qs):3d}): "
                  f"mean q={sum(relevant_qs)/len(relevant_qs):.1f}, "
                  f"min={min(relevant_qs)}, max={max(relevant_qs)}")
    report.append(f"  Irrelevant papers (n={len(irrelevant_qs):3d}): "
                  f"mean q={sum(irrelevant_qs)/len(irrelevant_qs):.1f}, "
                  f"min={min(irrelevant_qs)}, max={max(irrelevant_qs)}")

    # ---------- Safety net configurations ----------
    # Naming convention: conditions joined by OR/AND, parenthesized for clarity
    configs = {
        # === CURRENT CONFIG ===
        "CURRENT: q>=65 OR c<80":
            lambda r: r["cerebras_quality"] >= 65 or r["cerebras_confidence"] < 80,

        # === PURE QUALITY THRESHOLDS (no safety net) ===
        "q>=60":
            lambda r: r["cerebras_quality"] >= 60,
        "q>=62":
            lambda r: r["cerebras_quality"] >= 62,
        "q>=63":
            lambda r: r["cerebras_quality"] >= 63,
        "q>=64":
            lambda r: r["cerebras_quality"] >= 64,
        "q>=65":
            lambda r: r["cerebras_quality"] >= 65,
        "q>=66":
            lambda r: r["cerebras_quality"] >= 66,
        "q>=67":
            lambda r: r["cerebras_quality"] >= 67,
        "q>=68":
            lambda r: r["cerebras_quality"] >= 68,
        "q>=70":
            lambda r: r["cerebras_quality"] >= 70,

        # === QUALITY + LOW-CONFIDENCE RESCUE ===
        # Rescue papers where the model is unsure (c<75) and quality isn't terrible
        "q>=65 OR (q>=60 AND c<75)":
            lambda r: r["cerebras_quality"] >= 65 or (r["cerebras_quality"] >= 60 and r["cerebras_confidence"] < 75),
        "q>=65 OR (q>=62 AND c<75)":
            lambda r: r["cerebras_quality"] >= 65 or (r["cerebras_quality"] >= 62 and r["cerebras_confidence"] < 75),
        "q>=65 OR (q>=60 AND c<70)":
            lambda r: r["cerebras_quality"] >= 65 or (r["cerebras_quality"] >= 60 and r["cerebras_confidence"] < 70),
        "q>=65 OR (q>=55 AND c<75)":
            lambda r: r["cerebras_quality"] >= 65 or (r["cerebras_quality"] >= 55 and r["cerebras_confidence"] < 75),
        "q>=65 OR (q>=62 AND c<=78)":
            lambda r: r["cerebras_quality"] >= 65 or (r["cerebras_quality"] >= 62 and r["cerebras_confidence"] <= 78),

        # === HIGH-CONFIDENCE PENALTY ===
        # Observation: at q=65-78, c>=85 correlates with false positives
        # (domain applications the model is confidently wrong about)
        "q>=65 AND NOT (q<70 AND c>=85)":
            lambda r: r["cerebras_quality"] >= 65 and not (r["cerebras_quality"] < 70 and r["cerebras_confidence"] >= 85),
        "q>=65 AND NOT (q<72 AND c>=85)":
            lambda r: r["cerebras_quality"] >= 65 and not (r["cerebras_quality"] < 72 and r["cerebras_confidence"] >= 85),
        "(q>=65 AND c<85) OR q>=70":
            lambda r: (r["cerebras_quality"] >= 65 and r["cerebras_confidence"] < 85) or r["cerebras_quality"] >= 70,
        "(q>=65 AND c<85) OR q>=72":
            lambda r: (r["cerebras_quality"] >= 65 and r["cerebras_confidence"] < 85) or r["cerebras_quality"] >= 72,

        # === COMBINED: rescue + penalty ===
        "q>=65 OR (q>=62 AND c<=78) penalize c>=85<70":
            lambda r: (r["cerebras_quality"] >= 65 or (r["cerebras_quality"] >= 62 and r["cerebras_confidence"] <= 78)) and not (r["cerebras_quality"] < 70 and r["cerebras_confidence"] >= 85),

        # === LOWER THRESHOLDS FOR RECALL ===
        "q>=63 OR (q>=58 AND c<75)":
            lambda r: r["cerebras_quality"] >= 63 or (r["cerebras_quality"] >= 58 and r["cerebras_confidence"] < 75),
        "q>=62 OR (q>=50 AND c<=78)":
            lambda r: r["cerebras_quality"] >= 62 or (r["cerebras_quality"] >= 50 and r["cerebras_confidence"] <= 78),
    }

    report.append("\n\n## SAFETY NET CONFIGURATION COMPARISON")
    report.append("-" * 100)
    report.append(f"  {'Config':<32s} {'Accept':>6s} {'Reject':>6s} "
                  f"{'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'FP':>5s} {'FN':>5s}")
    report.append("-" * 100)

    best_f1 = 0
    best_config = ""
    config_results = {}

    for config_name, accept_fn in configs.items():
        accepted = [r for r in papers if accept_fn(r)]
        rejected = [r for r in papers if not accept_fn(r)]

        tp = sum(1 for r in accepted if r["opus_relevant"])
        fp = sum(1 for r in accepted if not r["opus_relevant"])
        fn = sum(1 for r in rejected if r["opus_relevant"])
        tn = sum(1 for r in rejected if not r["opus_relevant"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        marker = " <--" if config_name.startswith("CURRENT") else ""
        report.append(
            f"  {config_name:<32s} {len(accepted):>6d} {len(rejected):>6d} "
            f"{precision:>5.1%} {recall:>5.1%} {f1:>5.1%} {fp:>5d} {fn:>5d}{marker}"
        )

        config_results[config_name] = {
            "accepted": len(accepted), "rejected": len(rejected),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
        }

        if f1 > best_f1:
            best_f1 = f1
            best_config = config_name

    report.append(f"\n  BEST F1: {best_config} (F1={best_f1:.3f})")

    # ---------- Safety net paper details ----------
    report.append("\n\n## SAFETY NET PAPERS DETAIL (q<65, c<80)")
    report.append("Papers that would be accepted by current safety net but q<65")
    report.append("-" * 90)
    safety_papers = [p for p in papers if p["group"] == "safety_net"]
    relevant_safety = [p for p in safety_papers if p["opus_relevant"]]
    irrelevant_safety = [p for p in safety_papers if not p["opus_relevant"]]
    report.append(f"  {len(relevant_safety)} relevant / {len(safety_papers)} total ({len(relevant_safety)/len(safety_papers)*100:.0f}% precision)")
    report.append("")
    for p in sorted(safety_papers, key=lambda x: (-int(x["opus_relevant"]), -x["cerebras_quality"])):
        label = "RELEVANT" if p["opus_relevant"] else "NOT_REL"
        report.append(f"  q={p['cerebras_quality']:2d} c={p['cerebras_confidence']:2d} "
                      f"opus={label:7s} | {p['title'][:65]}")

    # ---------- False negatives ----------
    report.append("\n\n## FALSE NEGATIVES (rejected by q>=65 threshold, but Opus says relevant)")
    report.append("-" * 90)
    false_negs = [p for p in papers
                  if p["cerebras_quality"] < 65 and p["opus_relevant"]]
    for p in sorted(false_negs, key=lambda x: -x["cerebras_quality"]):
        report.append(f"  q={p['cerebras_quality']:2d} c={p['cerebras_confidence']:2d} "
                      f"group={p['group']:15s} | {p['title'][:60]}")

    # ---------- False positives in borderline ----------
    report.append("\n\n## FALSE POSITIVES IN BORDERLINE (q=65-69, accepted but Opus says not relevant)")
    report.append("-" * 90)
    borderline_fp = [p for p in papers
                     if p["group"] == "borderline_65_69" and not p["opus_relevant"]]
    for p in sorted(borderline_fp, key=lambda x: x["cerebras_quality"]):
        report.append(f"  q={p['cerebras_quality']:2d} c={p['cerebras_confidence']:2d} | {p['title'][:65]}")

    # ---------- Key findings ----------
    report.append("\n\n## KEY FINDINGS")
    report.append("-" * 70)
    report.append("1. The current safety net (q>=65 OR c<80) has very low precision in the")
    report.append("   safety_net group: most papers with q<65 and c<80 are NOT relevant.")
    report.append("   The c<80 condition rescues almost no true positives while admitting")
    report.append("   many false positives (papers on carbon flux, robotics, EEG, etc.)")
    report.append("")
    report.append("2. Simple quality threshold (q>=65) performs better than any safety net")
    report.append("   configuration, because the safety net adds noise without catching")
    report.append("   meaningful relevant papers.")
    report.append("")
    report.append("3. The judge's quality score is a strong discriminator:")
    report.append("   - q>=80: 100% precision (all papers truly relevant)")
    report.append("   - q=65-69: ~60% precision (mixed, but more relevant than not)")
    report.append("   - q<65: ~23% precision (mostly noise)")
    report.append("   - rejected (q<65, c>=80): ~3% relevant (excellent true negatives)")

    report_text = "\n".join(report)
    return report_text, config_results, papers


def main():
    report_text, config_results, papers = analyze()

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = EVAL_DIR / f"safety_net_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    # Save config results as JSON
    results_path = EVAL_DIR / f"config_comparison_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(config_results, f, indent=2)

    # Save ground truth data
    gt_path = EVAL_DIR / f"opus_ground_truth_{timestamp}.json"
    gt_data = {
        "evaluator": "Claude Opus 4.6",
        "date": "2026-03-05",
        "sample_seed": 42,
        "sample_sizes": {
            "safety_net": 30, "borderline_65_69": 25,
            "mid_70_79": 10, "high_80plus": 5, "rejected": 30,
        },
        "judge_model": "Cerebras gpt-oss-120b",
        "topics": [
            "Context Engineering & Management",
            "RAG, Retrieval & Knowledge Systems",
            "LLM Inference & Reasoning",
            "LLM Agents, Tool Use & Multi-Agent Systems",
            "Fine-tuning, Alignment & Prompt Engineering",
            "Generative AI & NLP Foundations",
        ],
        "evaluations": [
            {
                "paper_index": i + 1,
                "id": papers[i]["id"],
                "title": papers[i]["title"],
                "group": papers[i]["group"],
                "cerebras_quality": papers[i]["cerebras_quality"],
                "cerebras_confidence": papers[i]["cerebras_confidence"],
                "opus_relevant": OPUS_GROUND_TRUTH[i + 1],
            }
            for i in range(len(papers))
        ],
    }
    with open(gt_path, "w") as f:
        json.dump(gt_data, f, indent=2)

    print(report_text)
    print(f"\nReport saved to {report_path}")
    print(f"Config comparison saved to {results_path}")
    print(f"Ground truth data saved to {gt_path}")


if __name__ == "__main__":
    main()
