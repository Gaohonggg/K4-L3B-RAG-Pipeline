"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Không so sánh threshold với RRF score vì hai thang đo khác nhau.
"""

import logging
import os

from dotenv import load_dotenv

from .contracts import validate_search_results
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


load_dotenv()
logger = logging.getLogger(__name__)

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD") or "0.3")
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Return hybrid, dense-only, or PageIndex SearchResult items.

    ``use_reranking=False`` is the dense-only baseline for A/B evaluation.
    """
    if not isinstance(query, str) or not query.strip() or top_k < 1:
        return []
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1")

    dense = semantic_search(query, top_k=top_k * 2)
    validate_search_results(dense, top_k=top_k * 2, expected_method="dense")

    if use_reranking:
        sparse = lexical_search(query, top_k=top_k * 2)
        validate_search_results(sparse, top_k=top_k * 2, expected_method="bm25")
        ranked = rerank_rrf([dense, sparse], top_k=top_k)
        validate_search_results(ranked, top_k=top_k, expected_method="hybrid")
    else:
        ranked = dense[:top_k]

    best_dense_score = dense[0]["score"] if dense else 0.0
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            validate_search_results(
                fallback, top_k=top_k, expected_method="pageindex"
            )
            if fallback:
                return fallback
        except Exception as exc:
            logger.warning("PageIndex fallback unavailable: %s", exc)

    return ranked


if __name__ == "__main__":
    for result in retrieve("test query", top_k=3):
        print(result)
