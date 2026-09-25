from src.contracts import validate_search_results


def _result(item_id: str, score: float) -> dict:
    return {
        "id": item_id,
        "content": f"Evidence from {item_id}",
        "score": score,
        "metadata": {
            "source": "news/article.md",
            "title": "Article",
            "doc_type": "news",
            "url": "https://example.com/article",
            "chunk_index": int(item_id.rsplit("-", 1)[-1]),
        },
        "retrieval_method": "hybrid",
    }


def test_llm_reranker_validates_ids_and_appends_omissions(monkeypatch):
    import src.task11_llm_reranking as reranker

    candidates = [
        _result("chunk-0", 0.03),
        _result("chunk-1", 0.02),
        _result("chunk-2", 0.01),
    ]
    monkeypatch.setattr(
        reranker,
        "_request_ranking",
        lambda query, items: ["chunk-2", "invented", "chunk-2"],
    )

    results = reranker.rerank_with_llm("question", candidates, top_k=3)

    assert [item["id"] for item in results] == ["chunk-2", "chunk-0", "chunk-1"]
    validate_search_results(results, top_k=3, expected_method="hybrid")


def test_llm_reranker_falls_back_to_rrf_order(monkeypatch):
    import src.task11_llm_reranking as reranker

    candidates = [_result("chunk-0", 0.03), _result("chunk-1", 0.02)]

    def unavailable(query, items):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(reranker, "_request_ranking", unavailable)
    assert reranker.rerank_with_llm("question", candidates, top_k=1) == candidates[:1]


def test_highlighting_maps_citations_and_escapes_source_html():
    from src.ui_highlighting import cited_source_numbers, highlight_evidence

    answer = "Cầu có 12 nhịp [S2]."
    content = "<script>alert('x')</script> Cầu Trường Tiền có 12 nhịp."

    highlighted = highlight_evidence(content, answer, "Cầu có bao nhiêu nhịp?")

    assert cited_source_numbers(answer) == {2}
    assert "<script>" not in highlighted
    assert "&lt;script&gt;" in highlighted
    assert "<mark>12</mark>" not in highlighted  # Short numeric noise is not highlighted.
    assert "<mark>nhịp</mark>" in highlighted
