# Báo cáo đóng góp cá nhân

## Thông tin

- Họ và tên: Nguyễn Tất Đạt
- Mã học viên: 2A202602578
- Nhóm: K4-L3B
- Repository: `https://github.com/Gaohonggg/K4-L3B-RAG-Pipeline`
- Branch: `main`
- Git author dùng để đối chiếu: `Nguyen Dat/ Gaohonggg <dat111104@gmail.com>`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit | Trạng thái |
| --- | --- | --- | --- |
| Task 2 — dữ liệu news | Chạy crawler và lưu 5 bài viết JSON có `url`, `title`, `date_crawled`, `content_markdown`. | `877d85b`; `data/landing/news/article_01.json` đến `article_05.json` | Done |
| Task 3 — chuẩn hóa | Sửa và chạy lại luồng chuyển đổi, bổ sung validation cho legal/news input và làm sạch output Markdown. | `e00af37`; `src/task3_convert_markdown.py` | Done |
| Task 4 — chunking/indexing lần đầu | Triển khai load document, chunking, embedding và ghi vào ChromaDB theo module contract. | `aa41218`; `src/task4_chunking_indexing.py` | Done |
| Task 4 — tích hợp corpus | Chạy lại Task 2, tạo standardized corpus và cập nhật Task 4 để index dữ liệu thực tế. | `f00e1c1`; `data/standardized/**`, `src/task4_chunking_indexing.py` | Done |
| Task 4 — vector database | Hoàn thiện và chạy indexing, ghi persistent Chroma artifacts để pipeline retrieval có thể tái sử dụng. | `a5ba135`; `chroma_db/**` | Done |

Lịch sử Git ghi nhận 5 commit của tôi: `aa41218`, `877d85b`, `f00e1c1`, `e00af37`, `a5ba135`.

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Dùng `RecursiveCharacterTextSplitter` với `CHUNK_SIZE=800` và `CHUNK_OVERLAP=80`.
   
   **Lý do/evidence:** Corpus gồm các điều khoản pháp lý và bài viết tiếng Việt; chunk 800 ký tự giữ được phần lớn đoạn văn, overlap 10% hạn chế mất evidence tại ranh giới. Stable ID `<document-path>::chunk-<index>` giúp chạy lại không tạo bản ghi trùng.
   
   **Trade-off:** Chunk lớn giữ nhiều ngữ cảnh nhưng có thể làm giảm context precision; cần dùng evaluation để hiệu chỉnh thay vì chọn theo cảm tính.

2. **Quyết định:** Dùng cùng OpenAI embedding model cho indexing và dense query, lưu ChromaDB với cosine distance và metadata của embedding configuration.
   
   **Lý do/evidence:** Task 4 và Task 5 phải nằm trong cùng vector space; collection kiểm tra provider/model để không vô tình query một index không tương thích.
   
   **Trade-off:** Phụ thuộc API, network và chi phí embedding; đổi model yêu cầu re-index corpus.

## Kiểm thử và kết quả

- Đã chạy lại toàn bộ test trong `.venv` ngày 2026-09-26: `23 passed`.
- Acceptance tests xác nhận đủ 3 legal documents, 5 news JSON, standardized output và 15 golden Q&A.
- Persistent Chroma snapshot hiện có 106 indexed chunks, metadata ghi `openai`, `text-embedding-3-small`, `cosine`.
- A/B chính cho thấy dense-only đạt trung bình 0.6596, cao hơn hybrid+RRF 0.6224; kết quả này được báo cáo trung thực thay vì mặc định hybrid luôn tốt hơn.
- Bonus LLM reranker cải thiện MRR từ 0.9222 lên 1.0000 và nDCG@5 từ 0.9260 lên 0.9885 trên cùng 15 golden questions.

## Điều còn hạn chế

- Ba tài liệu pháp lý cần thêm manifest URL nguồn chính thức để provenance rõ hơn.
- Embedding pipeline hiện chỉ chấp nhận OpenAI; chưa có local embedding fallback.
- False-refusal rate của generation còn cao ở một số câu hỏi số liệu.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện là bổ sung source manifest, sau đó chạy grid experiment cho chunk size/overlap thay vì chỉ dùng một cấu hình.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh các phần việc có thể đối chiếu trong lịch sử Git và tôi có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-26
- Tên thành viên: Nguyễn Tất Đạt
