# Kết quả đánh giá hệ thống RAG

## Thông tin lần chạy

| Trường | Giá trị |
| --- | --- |
| Ngày chạy đánh giá chính | 2026-09-25 |
| Framework và phiên bản | RAGAS 0.4.3 |
| Evaluator model | gpt-4o-mini |
| Generator model | gpt-4o-mini |
| Embedding model | text-embedding-3-small |
| Phiên bản corpus/commit | 1db53eb + working-tree changes |
| Số mẫu golden dataset | 15 |
| `top_k` | 5 |
| Fallback threshold | `0.0`; tắt fallback cho cả hai nhánh A/B để cô lập retrieval strategy |

## Cấu hình đánh giá chính

- **Cấu hình A — dense-only:** OpenAI cosine retrieval, không dùng BM25 và RRF.
- **Cấu hình B — hybrid + RRF:** dense retrieval kết hợp BM25, fuse đúng một lần bằng reciprocal rank fusion.

Hai cấu hình dùng chung golden dataset, generator, evaluator, prompt, `top_k`, source selection và các bộ validator. Biến duy nhất thay đổi là retrieval strategy.

## Dữ liệu và quyết định indexing

Corpus gồm 8 tài liệu Markdown đã chuẩn hóa: 3 văn bản pháp lý/chính sách và 5 bài viết hoặc trang du lịch. Tiêu đề, loại tài liệu và đường dẫn nguồn nội bộ được giữ trong metadata xuyên suốt chunking, retrieval và citation rendering; URL gốc được giữ cho 5 tài liệu news. Ba tài liệu legal hiện chưa có URL nguồn chính thức và được ghi rõ ở phần hạn chế.

### Giải thích tham số Task 4

| Tham số | Lựa chọn | Lý do |
| --- | --- | --- |
| Chunking strategy | `RecursiveCharacterTextSplitter` | Ưu tiên ranh giới đoạn và dòng trước khi tách theo câu/từ, phù hợp với điều khoản pháp lý và các phần bài viết tiếng Việt. |
| `CHUNK_SIZE` | 800 ký tự | Giữ phần lớn một điều khoản hoặc đoạn bài viết trong cùng chunk nhưng vẫn đủ gọn cho retrieval. |
| `CHUNK_OVERLAP` | 80 ký tự (10%) | Giữ evidence gần ranh giới chunk mà không lặp quá nhiều nội dung. |
| Embedding | `text-embedding-3-small` | Indexing và dense query embedding dùng cùng provider/model, tránh vector-space mismatch. |
| Vector database | ChromaDB, cosine distance | Lưu bền vững cục bộ, ID ổn định và cosine score phù hợp cho fallback threshold. |
| Embedding batch | 64 texts/request | Giới hạn kích thước mỗi request và giữ đúng thứ tự response giữa các batch. |
| Chroma upsert batch | 100 records/write | Giới hạn kích thước mỗi lần ghi; stable chunk IDs cho phép upsert lặp lại và xóa stale records. |

Chunk ID có dạng `<relative-document-path>::chunk-<index>`, do đó chạy lại indexing không tạo bản ghi trùng. Collection lưu embedding provider/model trong metadata và từ chối index cũ không tương thích.

## Quyết định retrieval, generation và UI

- Dense search và BM25 cùng tuân theo `SearchResult`; RRF chỉ fuse một lần và không trộn trực tiếp thang điểm cosine với BM25.
- Fallback so sánh original dense cosine score với threshold. Lỗi PageIndex sẽ trả về hybrid ranking thay vì làm crash ứng dụng.
- Generation reorder evidence để giảm lost-in-the-middle, gán nhãn `[S1]`, `[S2]`, kiểm tra citation index và từ chối numeric claim không có trong evidence.
- Streamlit hiển thị answer, source, retrieval method, score, URL và retrieved passage.

## Điểm tổng hợp

| Metric | Cấu hình A | Cấu hình B | Delta B−A |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.7333 | 0.6667 | -0.0667 |
| Answer relevance | 0.1583 | 0.1504 | -0.0078 |
| Context recall | 0.9333 | 0.8667 | -0.0667 |
| Context precision | 0.8133 | 0.8059 | -0.0074 |
| **Trung bình** | **0.6596** | **0.6224** | **-0.0371** |

## So sánh A/B

- Cấu hình tốt hơn theo trung bình không trọng số của bốn metric: **Cấu hình A — dense-only**.
- Refusal rate: dense-only 26.7%; hybrid+RRF 33.3%.
- Mean end-to-end latency: dense-only 2.23 giây, hybrid+RRF 1.90 giây; B−A là -0.33 giây/query.
- Hybrid+RRF làm điểm trung bình thay đổi -0.0371; metric yếu nhất của nó là Answer relevance (0.1504).

## Các trường hợp kém nhất

| # | Câu hỏi | Cấu hình | Faithfulness | Relevance | Recall | Precision | Công đoạn lỗi | Nguyên nhân gốc |
| --: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | Lịch trình ngày thứ ba trong chuyến 3 ngày 2 đêm ở Huế gợi ý ba địa điểm nào? | hybrid_rrf | 0.0000 | 0.0000 | 0.0000 | 0.0000 | retrieval | Hybrid top-5 bỏ sót passage ngày thứ ba; các chunk mở đầu, listing và ngày thứ hai chiếm thứ hạng cao hơn. |
| 2 | Trong lịch trình 3 ngày 2 đêm ở Huế, giá vé tham quan cung An Định là bao nhiêu? | dense_only | 0.0000 | 0.0000 | 0.0000 | 1.0000 | generation | Đã retrieve các chunk liên quan nhưng generation hoặc citation validation tạo false refusal. |
| 3 | Theo quy chế, độ dày cát trung bình tối thiểu của bãi tắm biển là bao nhiêu? | dense_only | 0.0000 | 0.0000 | 1.0000 | 0.5000 | generation | Supporting context đã được retrieve nhưng grounded-answer validation vẫn trả refusal. |

## Đề xuất cải thiện

| Ưu tiên | Hành động | Evidence từ failure analysis | Tác động kỳ vọng | Cách kiểm chứng |
| ---: | --- | --- | --- | --- |
| 1 | Retrieve candidate pool lớn hơn trước khi chọn final top-k và dùng advanced reranker. | Câu lịch trình ngày thứ ba có recall/precision bằng 0 ở baseline hybrid. | Tăng context recall mà không thay generator. | Chạy lại cùng 15 golden questions và so sánh retrieval metrics. |
| 2 | Hiệu chỉnh refusal và citation validation tách biệt với retrieval. | Refusal rate là 26.7% và 33.3%, trong đó có trường hợp context đúng đã được retrieve. | Giảm false refusal nhưng không giảm faithfulness. | Bổ sung refusal-labelled cases và theo dõi false-refusal rate. |
| 3 | Hiệu chỉnh BM25/tokenization cho tiếng Việt. | Hybrid+RRF kém dense-only trên cả bốn aggregate metrics. | Cải thiện lexical ranking cho query paraphrase. | Theo dõi RRF rank theo từng case và yêu cầu A/B delta dương. |

## Thử nghiệm điểm cộng

### Advanced LLM reranker (+3)

Ngày 2026-09-26, thử nghiệm dùng cùng 15 golden questions, `candidate_k=15`, `top_k=5`, fallback threshold `0.0` và model `gpt-4o-mini`. Baseline là hybrid+RRF; nhánh thử nghiệm chỉ thêm LLM reranker trên cùng candidate pool.

| Retrieval metric | Hybrid + RRF | Hybrid + RRF + LLM reranker | Delta |
| --- | ---: | ---: | ---: |
| Source Hit@5 | 1.0000 | 1.0000 | +0.0000 |
| MRR | 0.9222 | 1.0000 | **+0.0778** |
| nDCG@5 | 0.9260 | 0.9885 | **+0.0625** |
| Context Precision@5 | 0.6267 | 0.6667 | **+0.0400** |
| Evidence token coverage | 0.9092 | 0.9741 | **+0.0649** |

Kết luận: LLM reranker giữ nguyên Hit@5 ở mức 100% và cải thiện thứ hạng, precision và coverage trên cùng candidate pool. Khi provider lỗi hoặc response không hợp lệ, module tự động giữ thứ tự RRF để không làm hỏng pipeline. Kết quả chi tiết nằm trong `group_project/evaluation/bonus_reranker_results.json`.

### Citation/source highlighting (+2)

- UI phân tích citation `[S1]`, `[S2]` từ answer và gắn nhãn “được trích dẫn” cho đúng source.
- Passage của cited source tô sáng các evidence terms giao nhau với answer/query bằng thẻ `<mark>`.
- Toàn bộ source content được HTML-escape trước khi render; unit test xác nhận script tag không thể được thực thi.
- Sidebar cho phép bật/tắt LLM reranker, hiển thị rõ retrieval method, score và reranker status.

Hai hạng mục trên cung cấp bằng chứng để đề nghị **+5 điểm bonus**. HyDE/query expansion và conversation memory không được claim.

## Hạn chế đã biết

- Hybrid+RRF thấp hơn dense-only 0.0371 điểm trung bình trong đánh giá chính; vì vậy không tuyên bố hybrid baseline là cải thiện.
- A/B chính tắt fallback để cô lập retrieval strategy; fallback behavior được kiểm tra riêng bằng contract tests.
- Live PageIndex fallback cần `PAGEINDEX_API_KEY`; provider absence được xử lý an toàn nhưng chưa phải live success demonstration.
- Answer relevance là aggregate metric yếu nhất và false refusal vẫn cần được hiệu chỉnh.
- LLM reranker tăng thêm chi phí và latency; do đó được thiết kế là tùy chọn trong UI.

## Kiểm thử

Chạy trong `.venv` ngày 2026-09-26:

```text
23 passed
```

Bao gồm contract tests, acceptance tests, LLM reranker fallback/ID validation và HTML-safe citation highlighting.

Các kiểm tra bổ sung đều đạt: `compileall`, `pip check`, Streamlit `AppTest` (2 titles, 1 chat input, 1 toggle) và health endpoint trả về `ok`. `git diff --check` không phát hiện lỗi whitespace; `.env` không nằm trong danh sách file được Git theo dõi.

## Tái lập kết quả

```bash
python -m group_project.evaluation.run_evaluation --top-k 5
python -m group_project.evaluation.run_bonus_reranker_evaluation \
  --top-k 5 --candidate-k 15
```

Per-case responses, source IDs, latency và bốn RAGAS scores được lưu trong `group_project/evaluation/evaluation_results.json`. Kết quả A/B bonus được lưu trong `group_project/evaluation/bonus_reranker_results.json`.
