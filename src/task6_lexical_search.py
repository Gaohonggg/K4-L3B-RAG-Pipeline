"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


CORPUS: list[dict] = []


from rank_bm25 import BM25Okapi
import numpy as np

def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    tokenized = [item["content"].lower().split() for item in corpus]
    return BM25Okapi(tokenized)

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    if not CORPUS:
        return []
    bm25 = build_bm25_index(CORPUS)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    
    # Bù đắp cho corpus siêu nhỏ (khiến IDF = 0.0) bằng term overlap
    q_set = set(query_tokens)
    for i, item in enumerate(CORPUS):
        overlap = len(q_set.intersection(item["content"].lower().split()))
        if overlap > 0:
            scores[i] += overlap * 1e-6

    indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for index in indices:
        if scores[index] <= 0:
            continue
        item = CORPUS[int(index)]
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[index]),
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        })
    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
