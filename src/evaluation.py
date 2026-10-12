"""Evaluation utilities: ranking metrics and TF-IDF vs Transformer comparison.

Run on the synthetic demo data:

    python -m src.evaluation --skip-transformer     # TF-IDF only, no model download
    python -m src.evaluation                         # also compares with all-MiniLM-L6-v2

CAUTION: the demo relevance grades are hand-assigned on 8 synthetic resumes. The
output is a sanity check of the code, not evidence of real-world accuracy.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from src.demo_data import DEMO_JOB_DESCRIPTION, DEMO_RESUMES, demo_relevance
from src.parser import ParsedDocument
from src.ranking import METHOD_TFIDF, METHOD_TRANSFORMER, RankingConfig, rank_resumes


def precision_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int, threshold: int = 1) -> float:
    """Fraction of the top-k ranked items whose grade is >= threshold."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = ranked_ids[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if relevance.get(item, 0) >= threshold) / len(top)


def dcg_at_k(grades: list[int], k: int) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    """Normalised discounted cumulative gain (graded relevance)."""
    if k <= 0:
        raise ValueError("k must be positive")
    actual = dcg_at_k([relevance.get(item, 0) for item in ranked_ids], k)
    ideal = dcg_at_k(sorted(relevance.values(), reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


def spearman_correlation(scores_a: list[float], scores_b: list[float]) -> float | None:
    """Rank correlation between two score lists (None if undefined)."""
    if len(scores_a) < 2 or len(scores_a) != len(scores_b):
        return None
    value = spearmanr(scores_a, scores_b).statistic
    return None if value != value else float(value)  # NaN check


def evaluate_ranking(ranked_ids: list[str], relevance: dict[str, int], k: int = 3) -> dict[str, float]:
    return {f"precision@{k}": precision_at_k(ranked_ids, relevance, k), f"ndcg@{k}": ndcg_at_k(ranked_ids, relevance, k)}


def compare_methods(
    job_description: str,
    documents: list[ParsedDocument],
    config: RankingConfig | None = None,
    matcher=None,
    include_transformer: bool = True,
) -> pd.DataFrame:
    """Rank with TF-IDF (and optionally the transformer) and return a side-by-side table."""
    tfidf = rank_resumes(job_description, documents, METHOD_TFIDF, config)
    table = pd.DataFrame(
        {
            "filename": [r.filename for r in tfidf],
            "tfidf_final": [r.final_score for r in tfidf],
            "tfidf_rank": [r.rank for r in tfidf],
            "skill_coverage": [r.skill_coverage for r in tfidf],
        }
    )
    if include_transformer:
        transformer = rank_resumes(job_description, documents, METHOD_TRANSFORMER, config, matcher=matcher)
        by_name = {r.filename: r for r in transformer}
        table["transformer_final"] = table["filename"].map(lambda n: by_name[n].final_score)
        table["transformer_rank"] = table["filename"].map(lambda n: by_name[n].rank)
        table["rank_change"] = table["tfidf_rank"] - table["transformer_rank"]
    return table


def _demo_documents() -> list[ParsedDocument]:
    return [ParsedDocument(filename=name, text=text) for name, (text, _) in DEMO_RESUMES.items()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate rankings on the synthetic demo set.")
    parser.add_argument("--skip-transformer", action="store_true", help="Do not load the transformer model.")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", default="reports/evaluation_demo.csv")
    args = parser.parse_args(argv)

    documents = _demo_documents()
    relevance = demo_relevance()
    try:
        table = compare_methods(DEMO_JOB_DESCRIPTION, documents, include_transformer=not args.skip_transformer)
    except Exception as exc:  # e.g. model cannot be downloaded offline
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Tip: re-run with --skip-transformer to evaluate the TF-IDF baseline only.", file=sys.stderr)
        return 1

    table["hand_assigned_grade"] = table["filename"].map(relevance)
    print("\nSide-by-side scores (synthetic demo data):")
    print(table.to_string(index=False))

    methods = [("tfidf", "tfidf_rank")] + ([] if args.skip_transformer else [("transformer", "transformer_rank")])
    print(f"\nRanking metrics against hand-assigned grades (k={args.k}) - sanity check only:")
    for name, rank_col in methods:
        ordered = table.sort_values(rank_col)["filename"].tolist()
        metrics = evaluate_ranking(ordered, relevance, args.k)
        print(f"  {name:12s} " + "  ".join(f"{key}={value:.3f}" for key, value in metrics.items()))

    if not args.skip_transformer:
        rho = spearman_correlation(table["tfidf_final"].tolist(), table["transformer_final"].tolist())
        print(f"\nSpearman correlation between TF-IDF and transformer final scores: {rho}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"\nSaved table to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
