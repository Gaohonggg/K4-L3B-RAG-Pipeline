"""Task 6 — BM25 over the same indexed chunks used by dense search."""

import re

import numpy as np
from rank_bm25 import BM25Okapi

from .contracts import validate_document
from .task4_chunking_indexing import get_collection


# Explicit corpus overrides are useful for small experiments and contract tests.
# In normal operation, the corpus is read from the persistent Chroma collection.
CORPUS: list[dict] = []
_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def _load_indexed_corpus() -> list[dict]:
    """Read all indexed chunks without requesting embeddings or calling APIs."""
    data = get_collection().get(include=["documents", "metadatas"])
    ids = data.get("ids") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    if not (len(ids) == len(documents) == len(metadatas)):
        raise ValueError("Chroma returned incomplete BM25 corpus fields")

    corpus: list[dict] = []
    for item_id, content, metadata in zip(ids, documents, metadatas):
        item = {
            "id": item_id,
            "content": content,
            "metadata": {**metadata, "url": metadata.get("url") or None},
        }
        validate_document(item, require_chunk=True)
        corpus.append(item)
    return sorted(corpus, key=lambda item: item["id"])

def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    tokenized = [_tokenize(item["content"]) for item in corpus]
    return BM25Okapi(tokenized)

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    if not isinstance(query, str) or not query.strip() or top_k < 1:
        return []
    corpus = CORPUS if CORPUS else _load_indexed_corpus()
    if not corpus:
        return []
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    bm25 = build_bm25_index(corpus)
    scores = bm25.get_scores(query_tokens)
    # For tiny corpora, an exact token overlap breaks zero-IDF ties.
    q_set = set(query_tokens)
    for i, item in enumerate(corpus):
        overlap = len(q_set.intersection(_tokenize(item["content"])))
        if overlap > 0:
            scores[i] += overlap * 1e-6

    indices = np.argsort(-scores, kind="stable")[:top_k]
    results: list[dict] = []
    for index in indices:
        if scores[index] <= 0:
            continue
        item = corpus[int(index)]
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
