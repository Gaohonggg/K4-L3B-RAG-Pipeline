"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import argparse
import json
import logging
import os
import re
import sys

from dotenv import load_dotenv

from .contracts import validate_generation_result, validate_search_results
from .task9_retrieval_pipeline import retrieve


load_dotenv()
logger = logging.getLogger(__name__)

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 1024
SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

# SYSTEM_PROMPT gốc: Trả lời chỉ từ context được cung cấp.
# Mỗi khẳng định phải có citation. Nếu thiếu evidence, hãy từ chối xác minh.
SYSTEM_PROMPT = (
    "Chỉ trả lời bằng thông tin có trong các đoạn nguồn được cung cấp. "
    "Trả lời đúng thuộc tính được hỏi, ngắn gọn. Nếu câu hỏi hỏi một con số, "
    "chỉ nêu con số và đơn vị của thuộc tính đó, không thêm các con số khác. "
    "Giữ nguyên số liệu, đơn vị, thuật ngữ và tên riêng từ nguồn; "
    "không ghép hai số hoặc hai thuật ngữ khác nhau thành một khẳng định. "
    "Mỗi khẳng định thực tế phải kèm nhãn nguồn chính xác dạng [S1], [S2], ... "
    "Không dùng nhãn không có trong context. Nếu nguồn không đủ để trả lời, "
    f"chỉ trả lời: {SAFE_REFUSAL}"
)
_CITATION = re.compile(r"\[S(\d+)\]")
_ANSWER_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_QUANTITY_TARGET = re.compile(r"bao\s+nhiêu\s+(\w+)", re.IGNORECASE)


def _select_sources(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Retain exact numeric evidence for quantity questions when available."""
    selected = list(candidates[:top_k])
    match = _QUANTITY_TARGET.search(query)
    if not match or len(candidates) <= top_k or not selected:
        return selected

    target = match.group(1).casefold()
    numeric_fact = re.compile(rf"\b\d+(?:[.,]\d+)?\s+{re.escape(target)}\b")
    if any(numeric_fact.search(item["content"].casefold()) for item in selected):
        return selected
    for item in candidates[top_k:]:
        if numeric_fact.search(item["content"].casefold()):
            selected[-1] = item
            break
    return selected


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context, không làm mất ID."""
    if len(chunks) <= 2:
        return list(chunks)
    return chunks[::2] + chunks[1::2][::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label ổn định."""
    parts: list[str] = []
    for position, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        label = chunk.get("_citation_label", f"S{position}")
        header = f"[{label}] Title: {metadata['title']} | Source: {metadata['source']}"
        if metadata.get("url"):
            header += f" | URL: {metadata['url']}"
        parts.append(f"{header}\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    model = os.getenv("LLM_MODEL", "").strip()
    if not model:
        raise ValueError("Set LLM_MODEL in .env")

    if provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY in .env")
        response = OpenAI(api_key=api_key, timeout=30).chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        answer = response.choices[0].message.content
    elif provider == "gemini":
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY in .env")
        client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=30_000)
        )
        response = client.models.generate_content(
            model=model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
            ),
        )
        answer = response.text
    elif provider == "anthropic":
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Set ANTHROPIC_API_KEY in .env")
        response = Anthropic(api_key=api_key, timeout=30).messages.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        answer = "".join(
            block.text for block in response.content if block.type == "text"
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("LLM returned an empty answer")
    return answer.strip()


def _refusal() -> dict:
    return {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}


def generate_grounded_answer(
    query: str,
    candidates: list[dict],
    top_k: int = TOP_K,
) -> tuple[str, list[dict]]:
    """Generate and validate one answer from pre-retrieved candidates.

    Keeping generation separate from retrieval lets the evaluation compare retrieval
    strategies while holding the prompt, model, source selection and validators fixed.
    """
    if not isinstance(query, str) or not query.strip() or top_k < 1:
        return SAFE_REFUSAL, []

    try:
        validate_search_results(candidates)
        if not candidates:
            logger.info("Không tìm thấy nguồn phù hợp cho câu hỏi.")
            return SAFE_REFUSAL, []

        sources = _select_sources(query, candidates, top_k)
        validate_search_results(sources, top_k=top_k)
        labeled = [
            {**source, "_citation_label": f"S{index}"}
            for index, source in enumerate(sources, 1)
        ]
        context = format_context(reorder_for_llm(labeled))
        answer = call_llm(
            SYSTEM_PROMPT,
            f"Context:\n{context}\n\nQuestion: {query.strip()}",
        )
    except Exception as exc:
        logger.warning("Could not generate grounded answer: %s", exc)
        return SAFE_REFUSAL, []

    citations = {int(index) for index in _CITATION.findall(answer)}
    if answer == SAFE_REFUSAL or not citations or any(
        index < 1 or index > len(sources) for index in citations
    ):
        logger.info("LLM từ chối hoặc câu trả lời không có citation hợp lệ.")
        return SAFE_REFUSAL, []

    cited_text = "\n".join(sources[index - 1]["content"] for index in citations)
    numbers = _ANSWER_NUMBER.findall(_CITATION.sub("", answer))
    cited_numbers = set(_ANSWER_NUMBER.findall(cited_text))
    if any(number not in cited_numbers for number in numbers):
        logger.warning("Citation không chứa số liệu được nêu trong câu trả lời.")
        return SAFE_REFUSAL, []

    return answer, sources


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult với citation đối chiếu được với sources."""
    if not isinstance(query, str) or not query.strip() or top_k < 1:
        return _refusal()

    try:
        candidate_k = max(top_k, min(top_k * 2, 20))
        candidates = retrieve(query, top_k=candidate_k)
        validate_search_results(candidates, top_k=candidate_k)
    except Exception as exc:
        logger.warning("Could not retrieve evidence: %s", exc)
        return _refusal()

    answer, sources = generate_grounded_answer(query, candidates, top_k=top_k)
    if not sources:
        return _refusal()

    result = {
        "answer": answer,
        "sources": sources,
        "retrieval_source": (
            "pageindex"
            if sources[0]["retrieval_method"] == "pageindex"
            else "hybrid"
        ),
    }
    validate_generation_result(result)
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description="Hỏi chatbot RAG với citation.")
    parser.add_argument("--query", help="Câu hỏi cần trả lời")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Số nguồn tối đa")
    args = parser.parse_args()
    if args.query:
        query = args.query
    elif sys.stdin.isatty():
        query = input("Nhập câu hỏi: ").strip()
    else:
        parser.print_help()
        return 2
    if not query:
        parser.error("Câu hỏi không được rỗng")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print(json.dumps(generate_with_citation(query, top_k=args.top_k), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
