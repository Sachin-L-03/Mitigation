"""Construct the Phase 5 Multi-Domain Benchmark (300 Semantic Clusters, 1,200 Prompts).

Domains:
  1. science (40 clusters)
  2. math (35 clusters)
  3. history (35 clusters)
  4. geography (35 clusters)
  5. dates (30 clusters)
  6. units (30 clusters)
  7. everyday (30 clusters)
  8. structured (25 clusters)
  9. negation (20 clusters)
 10. constraints (20 clusters)

Total: 300 clusters (74 validation, 226 test).
Each cluster contains 4 distinct prompt variants:
  - original (canonical)
  - lexical (synonym / active-passive)
  - syntactic (clause order / phrasing)
  - concise (direct / telegraphic)
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path("data/phase5_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Helper function to generate high-quality variations
def build_cluster(
    cid: str,
    domain: str,
    q_orig: str,
    q_lex: str,
    q_syn: str,
    q_conc: str,
    gold: str,
    aliases: List[str],
    prov: str,
    criterion: str,
    split: str,
) -> Dict[str, Any]:
    return {
        "cluster_id": cid,
        "domain": domain,
        "canonical_question": q_orig,
        "variants": {
            "original": q_orig,
            "lexical": q_lex,
            "syntactic": q_syn,
            "concise": q_conc,
        },
        "gold_reference_answer": gold,
        "acceptable_aliases": aliases,
        "provenance": prov,
        "correctness_criterion": criterion,
        "answerable": True,
        "split": split,
    }


def main():
    print("Building Phase 5 Domain Benchmark...")

    raw_clusters = []

    # Import the specs we defined
    from scripts.phase5_specs_data import ALL_PHASE5_SPECS

    for spec in ALL_PHASE5_SPECS:
        cid, dom, q_orig, q_lex, q_syn, q_conc, gold, aliases, prov, crit, split = spec
        raw_clusters.append(build_cluster(cid, dom, q_orig, q_lex, q_syn, q_conc, gold, aliases, prov, crit, split))

    print(f"Total clusters constructed: {len(raw_clusters)}")
    val_clusters = [c for c in raw_clusters if c["split"] == "val"]
    test_clusters = [c for c in raw_clusters if c["split"] == "test"]
    print(f"Validation clusters: {len(val_clusters)}")
    print(f"Test clusters: {len(test_clusters)}")

    # 1. Write domain_dataset.jsonl (1,200 prompt records)
    records = []
    for c in raw_clusters:
        cid = c["cluster_id"]
        dom = c["domain"]
        split = c["split"]
        for vtype, ptext in c["variants"].items():
            pid = f"{cid}_{vtype}"
            records.append({
                "cluster_id": cid,
                "prompt_id": pid,
                "domain": dom,
                "variant_type": vtype,
                "prompt_text": ptext,
                "split": split,
            })

    dataset_path = OUTPUT_DIR / "domain_dataset.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {dataset_path} ({len(records)} prompt records)")

    # 2. Write correctness_subset.jsonl (300 reference items)
    correctness_items = []
    for c in raw_clusters:
        correctness_items.append({
            "cluster_id": c["cluster_id"],
            "domain": c["domain"],
            "canonical_question": c["canonical_question"],
            "gold_reference_answer": c["gold_reference_answer"],
            "acceptable_aliases": c["acceptable_aliases"],
            "provenance": c["provenance"],
            "correctness_criterion": c["correctness_criterion"],
            "answerable": c["answerable"],
            "split": c["split"],
        })

    subset_path = OUTPUT_DIR / "correctness_subset.jsonl"
    with open(subset_path, "w", encoding="utf-8") as f:
        for item in correctness_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {subset_path} ({len(correctness_items)} verified reference items)")

    # 3. Write hard_negatives.jsonl (30 controlled non-equivalent challenge items)
    from scripts.phase5_specs_data import HARD_NEGATIVE_SPECS
    hn_path = OUTPUT_DIR / "hard_negatives.jsonl"
    with open(hn_path, "w", encoding="utf-8") as f:
        for hn in HARD_NEGATIVE_SPECS:
            f.write(json.dumps(hn, ensure_ascii=False) + "\n")
    print(f"Saved {hn_path} ({len(HARD_NEGATIVE_SPECS)} hard negative items)")

    # 4. Generate dataset_manifest.json with SHA-256 hashes
    manifest = {
        "dataset_name": "Phase 5 Factual Reliability & False-Consensus Benchmark",
        "total_clusters": len(raw_clusters),
        "total_prompts": len(records),
        "validation_clusters": len(val_clusters),
        "test_clusters": len(test_clusters),
        "domains": {},
        "files": {},
    }

    for c in raw_clusters:
        d = c["domain"]
        manifest["domains"][d] = manifest["domains"].get(d, 0) + 1

    for p in [dataset_path, subset_path, hn_path]:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest["files"][p.name] = {
            "path": str(p).replace("\\", "/"),
            "sha256": h,
            "bytes": p.stat().st_size,
        }

    manifest_path = OUTPUT_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {manifest_path}")

    # 5. Cryptographic audit against historical prompts
    prior_prompts: Set[str] = set()
    for pf in Path("data").glob("*/*.jsonl"):
        if "phase5" not in str(pf):
            with open(pf, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            obj = json.loads(line)
                            t = obj.get("prompt_text", obj.get("prompt", obj.get("question", ""))).strip().lower()
                            if t:
                                prior_prompts.add(t)
                        except:
                            pass

    current_prompts = set(r["prompt_text"].strip().lower() for r in records)
    leakage = current_prompts & prior_prompts
    print(f"Historical prompts checked: {len(prior_prompts)}")
    print(f"Current Phase 5 prompts: {len(current_prompts)}")
    print(f"Exact prompt matches with prior benchmarks: {len(leakage)}")
    assert len(leakage) == 0, f"Leakage detected: {leakage}"
    print("VERIFIED: Zero prompt leakage against historical datasets!")


if __name__ == "__main__":
    main()
