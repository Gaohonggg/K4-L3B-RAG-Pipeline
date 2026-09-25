## 1. Chọn đề tài

- Chọn một chủ đề trong [danh sách gợi ý](SUGGESTED_TOPICS.md) hoặc chủ đề khác.
- Phân công role, chia nhiệm vụ các thành viên
- Mỗi thành viên ghi lại commit mình phụ trách để hoàn thiện individual report

## 2. Cài môi trường

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m playwright install chromium
cp .env.example .env
```

## 3. Thu thập dữ liệu

- Tải tối thiểu 3 PDF/DOCX vào `data/landing/legal/`.
- Crawl tối thiểu 5 bài vào `data/landing/news/`.
- Mỗi JSON có `url`, `title`, `date_crawled`, `content_markdown`.

```bash
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
```

Trong repo có setup sẵn Crawl4AI, các bạn tùy ý sử dụng công cụ khác của mình

## 4. Chuẩn hóa Markdown

Hoàn thiện Task 3 rồi chạy:

```bash
python -m src.task3_convert_markdown
```

Trong repo có setup sẵn marktidown, các bạn tùy ý sử dụng công cụ khác

## 5. Chunk, embedding và index

```bash
python -m src.task4_chunking_indexing
```

## 6. Xây dựng hybrid retrieval

- Task 5: semantic search từ ChromaDB.
- Task 6: BM25 trên cùng corpus chunks.
- Task 7: RRF gộp hai bảng xếp hạng theo ID.

```bash
python -m src.task5_semantic_search
python -m src.task6_lexical_search
python -m src.task7_reranking
```

Về rerank là không bắt buộc, các bận có thể sử dụng Jina, hoặc tự self host BGE (hoặc không làm)

## 7. Hoàn thiện fallback và retrieval pipeline

- Task 8 trả `retrieval_method="pageindex"`.
- Task 9 chỉ chạy RRF một lần.
- Calibrate threshold bằng query đúng domain và query ngoài domain.
- Dùng dense cosine score gốc để quyết định fallback.

## 8. Generation có citation

Hoàn thiện Task 10:

- Reorder chunks nhưng không làm mất ID.
- Context có title/source.
- Dispatch theo `LLM_PROVIDER`: OpenAI, Gemini hoặc Anthropic Claude.
- Không đủ evidence thì trả safe refusal.

```bash
python -m src.task10_generation
```

**Hoàn thành khi:** answer đúng `GenerationResult` và citation map được về `sources`.

## 9. Chatbot và evaluation

```bash
streamlit run app.py
```

- UI hiển thị answer, source, retrieval method và score.
- Tạo ít nhất 15 golden Q&A dựa trên corpus.
- Chạy 4 metric: faithfulness, answer relevance, context recall, context precision.
- So sánh dense-only với hybrid + RRF trên cùng cấu hình còn lại.
- Điền `reports/RESULT.md`.

**Hoàn thành khi:** chatbot chạy end-to-end và báo cáo không còn placeholder.

## 10. Kiểm tra và nộp bài

```bash
pytest tests/test_contracts.py -q
pytest tests/test_acceptance.py -q
pytest -q
```

- Mỗi thành viên hoàn thiện individual report.
- Kiểm tra repository không chứa `.env`, API key hoặc file cache.
- Demo một query đúng, một query ngoài domain và kết quả A/B.
