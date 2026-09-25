"""Streamlit UI for the grounded Hue tourism RAG chatbot."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation
from src.task11_llm_reranking import generate_with_llm_reranking
from src.ui_highlighting import cited_source_numbers, highlight_evidence


load_dotenv()

st.set_page_config(
    page_title="Hue Tourism RAG",
    page_icon="⛩️",
    layout="wide",
)


def render_sources(
    sources: list[dict],
    *,
    answer: str,
    query: str = "",
    advanced_reranking: bool = False,
) -> None:
    """Render retrieval evidence without hiding method or score."""
    if not sources:
        st.info("Câu trả lời này không sử dụng được nguồn nào trong corpus.")
        return

    cited_numbers = cited_source_numbers(answer)
    st.markdown("**Nguồn đã dùng**")
    for index, source in enumerate(sources, 1):
        metadata = source["metadata"]
        method = source["retrieval_method"]
        score = float(source["score"])
        title = metadata["title"]
        source_name = metadata["source"]

        cited_label = " · được trích dẫn" if index in cited_numbers else ""
        reranker_label = " · LLM reranker" if advanced_reranking else ""
        with st.expander(
            f"[S{index}] {title} · {method}{reranker_label} "
            f"· score={score:.4f}{cited_label}"
        ):
            st.caption(f"Tệp nguồn: {source_name}")
            if metadata.get("url"):
                st.link_button("Mở nguồn gốc", metadata["url"])
            if index in cited_numbers:
                st.caption("Các từ khóa liên quan được tô sáng trong đoạn bằng chứng.")
                st.markdown(
                    highlight_evidence(source["content"], answer, query),
                    unsafe_allow_html=True,
                )
            else:
                st.write(source["content"])


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Hue Tourism RAG")
    st.caption(
        "Chatbot hỏi đáp có trích dẫn từ văn bản du lịch và chính sách "
        "du lịch thành phố Huế."
    )
    top_k = st.slider("Số chunks truy xuất", min_value=3, max_value=10, value=5)
    advanced_reranking = st.toggle(
        "LLM reranker nâng cao",
        value=False,
        help="Rerank candidate pool sau hybrid + RRF; có thêm chi phí và độ trễ LLM.",
    )
    if st.button("Xóa lịch sử hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("Chatbot du lịch Huế")
st.caption(
    "Câu trả lời chỉ dựa trên corpus của nhóm. Mở phần nguồn để kiểm tra "
    "nội dung, phương pháp retrieval và score."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []),
                answer=message["content"],
                query=message.get("query", ""),
                advanced_reranking=message.get("advanced_reranking", False),
            )

query = st.chat_input("Hỏi về du lịch, văn hóa hoặc chính sách du lịch Huế...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất nguồn và tạo câu trả lời..."):
            generator = (
                generate_with_llm_reranking
                if advanced_reranking
                else generate_with_citation
            )
            result = generator(query, top_k=top_k)
        st.markdown(result["answer"])
        render_sources(
            result["sources"],
            answer=result["answer"],
            query=query,
            advanced_reranking=advanced_reranking,
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
            "query": query,
            "advanced_reranking": advanced_reranking,
        }
    )
