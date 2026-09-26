# Individual contribution report

---

## Thông tin

- Họ và tên: Hoàng Quốc Dũng
- Mã học viên: 2A202602523
- Nhóm: Team03
- Repository/branch: main

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Task 3 (Convert) | Viết script chuyển PDF/HTML sang Markdown | `src/task3_convert_markdown.py` | Done |
| Task 8 (PageIndex) | Xây dựng cơ chế fallback dùng vectorless search từ PageIndex | `src/task8_pageindex_vectorless.py` | Done |
| Task 9 (Pipeline) | Gom Dense, Sparse, RRF và Fallback thành 1 pipeline | `src/task9_retrieval_pipeline.py` | Done |
| Task 10 (Gen/UI) | Cài đặt Generation (citations) và giao diện Streamlit, Evaluation | `task10_generation.py`, `app.py`, `RESULT.md` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Viết fallback PageIndex chỉ kích hoạt khi điểm cosine của Dense Search dưới `SCORE_THRESHOLD`.
   **Lý do/evidence:** Hạn chế gọi API bên thứ 3 (PageIndex) giúp tối ưu thời gian phản hồi (latency), chỉ gọi khi thực sự bí (câu hỏi out-of-domain).
   **Trade-off:** Chịu thêm độ trễ khi fallback bị kích hoạt.

2. **Quyết định:** Bổ sung tính năng LLM Reranker (Bonus).
   **Lý do/evidence:** Muốn tối đa hoá Context Recall và Precision bằng LLM (gpt-4o-mini). Ragas metric cho thấy Reranker tăng MRR lên +0.0778.
   **Trade-off:** Đòi hỏi thêm chi phí API và tăng mạnh độ trễ (latency) của hệ thống.

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: `pytest tests/test_acceptance.py` và chạy script Evaluation `run_evaluation`.
- Kết quả trước/sau nếu có: Đã pass 100% test case (23/23). Report được sinh tự động.
- Lỗi đã phát hiện và cách xử lý: Xử lý an toàn lỗi API từ PageIndex (try-except) để pipeline không bị crash khi mạng rớt.

## Điều còn hạn chế

- Một hạn chế cụ thể của phần tôi làm: Việc render markdown cho UI (Streamlit) đôi lúc chưa highlight mượt mà được những câu văn dài bị ngắt dòng trong PDF.
- Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện: Tối ưu UI hiển thị citation và thêm tính năng chitchat memory.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 26/09/2026
- Tên thành viên: Hoàng Quốc Dũng
