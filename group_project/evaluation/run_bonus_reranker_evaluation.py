"""A/B retrieval evaluation for the optional LLM reranker bonus."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import statistics
from datetime import date
from pathlib import Path
from typing import Any

from src.task11_llm_reranking import rerank_with_llm
from src.task9_retrieval_pipeline import retrieve


ROOT_DIR = Path(__file__).resolve().parents[2]
EVALUATION_DIR = ROOT_DIR / "group_project" / "evaluation"
GOLDEN_DATASET_PATH = EVALUATION_DIR / "golden_dataset.json"
OUTPUT_PATH = EVALUATION_DIR / "bonus_reranker_results.json"
_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = {
    "bằng",
    "bao",
    "cho",
    "các",
    "có",
    "của",
    "là",
    "một",
    "những",
    "theo",
    "trong",
    "trên",
    "và",
    "với",
    "được",
}
logger = logging.getLogger(__name__)


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN.findall(text)
        if len(token) >= 3 and token.casefold() not in _STOPWORDS
    }


def _source_matches(item: dict, expected_source: str) -> bool:
    return item["metadata"]["source"].endswith(expected_source)


def _item_relevance(item: dict, expected_source: str, reference_tokens: set[str]) -> float:
    if not _source_matches(item, expected_source) or not reference_tokens:
        return 0.0
    return len(reference_tokens & _tokens(item["content"])) / len(reference_tokens)


def _score_ranking(
    ranking: list[dict],
    *,
    expected_source: str,
    expected_context: str,
) -> dict[str, float]:
    reference_tokens = _tokens(expected_context)
    combined_tokens = _tokens("\n".join(item["content"] for item in ranking))
    coverage = (
        len(reference_tokens & combined_tokens) / len(reference_tokens)
        if reference_tokens
        else 0.0
    )
    relevance = [
        _item_relevance(item, expected_source, reference_tokens) for item in ranking
    ]
    binary = [score >= 0.25 for score in relevance]
    first_rank = next((index for index, relevant in enumerate(binary, 1) if relevant), None)
    precision = sum(binary) / len(binary) if binary else 0.0
    dcg = sum(value / math.log2(index + 1) for index, value in enumerate(binary, 1))
    ideal_count = min(sum(binary), len(binary))
    ideal_dcg = sum(1 / math.log2(index + 1) for index in range(1, ideal_count + 1))
    return {
        "source_hit_at_k": float(any(binary)),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
        "context_precision_at_k": precision,
        "evidence_token_coverage": coverage,
    }


def _mean_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    metric_names = rows[0][key].keys()
    return {
        metric: statistics.fmean(row[key][metric] for row in rows)
        for metric in metric_names
    }


def run_bonus_evaluation(top_k: int = 5, candidate_k: int = 15) -> dict[str, Any]:
    dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(dataset, 1):
        logger.info("Reranker A/B %d/%d", index, len(dataset))
        candidates = retrieve(
            case["question"],
            top_k=candidate_k,
            score_threshold=0.0,
            use_reranking=True,
        )
        baseline = candidates[:top_k]
        advanced = rerank_with_llm(case["question"], candidates, top_k=top_k)
        rows.append(
            {
                "case_id": index,
                "question": case["question"],
                "expected_source": case["source"],
                "baseline_ids": [item["id"] for item in baseline],
                "llm_reranker_ids": [item["id"] for item in advanced],
                "baseline": _score_ranking(
                    baseline,
                    expected_source=case["source"],
                    expected_context=case["expected_context"],
                ),
                "llm_reranker": _score_ranking(
                    advanced,
                    expected_source=case["source"],
                    expected_context=case["expected_context"],
                ),
            }
        )

    baseline_summary = _mean_metrics(rows, "baseline")
    reranker_summary = _mean_metrics(rows, "llm_reranker")
    payload = {
        "metadata": {
            "evaluation_date": date.today().isoformat(),
            "dataset_size": len(dataset),
            "top_k": top_k,
            "candidate_k": candidate_k,
            "baseline": "hybrid + RRF",
            "experiment": "hybrid + RRF + LLM reranker",
            "reranker_model": os.getenv("LLM_MODEL", "").strip(),
            "fallback_threshold": 0.0,
        },
        "summary": {
            "baseline": baseline_summary,
            "llm_reranker": reranker_summary,
            "delta": {
                metric: reranker_summary[metric] - baseline_summary[metric]
                for metric in baseline_summary
            },
        },
        "rows": rows,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare RRF against RRF followed by an LLM reranker."
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=15)
    args = parser.parse_args()
    if args.top_k < 1 or args.candidate_k < args.top_k:
        parser.error("candidate-k must be greater than or equal to top-k")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_bonus_evaluation(args.top_k, args.candidate_k)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
