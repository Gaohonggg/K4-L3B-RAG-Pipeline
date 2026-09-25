"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.

-> Dùng Jina hoặc self host hoặc bất cứ công cụ nào bạn quen
"""


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):   # rank bắt đầu từ 1
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
            items[item_id] = item                      # giữ nguyên nội dung gốc

    ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    results = []
    for item_id in ranked_ids[:top_k]:
        result = items[item_id].copy()                 # không mutate dict gốc
        result["score"] = scores[item_id]              # RRF score (không dùng để fallback)
        result["retrieval_method"] = "hybrid"          # luôn đánh dấu là hybrid
        results.append(result)

    return results


if __name__ == "__main__":
    print("RRF reranker is ready. Run pytest to verify its contract.")
