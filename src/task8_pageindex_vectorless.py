"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .contracts import validate_search_results


ROOT_DIR = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = ROOT_DIR / "data" / "standardized"
CACHE_FILE = ROOT_DIR / "pageindex_doc_ids.json"
TEMP_PDF_DIR = ROOT_DIR / "data" / "_tmp_pdf"

load_dotenv(ROOT_DIR / ".env")
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()

DEFAULT_TIMEOUT = 15  # seconds
_SOURCE_LINE = re.compile(r"^\*\*(?:Source|URL):\*\*\s*(\S+)", re.IGNORECASE)

logger = logging.getLogger(__name__)


def _extract_metadata(md_path: Path) -> dict[str, Any]:
    """Trích xuất metadata từ file Markdown chuẩn hóa."""
    relative_path = md_path.relative_to(STANDARDIZED_DIR)
    content = md_path.read_text(encoding="utf-8").strip()

    title = md_path.stem
    heading_found = False
    url: str | None = None

    for line in content.splitlines()[:30]:
        stripped = line.strip()
        if stripped.startswith("# ") and not heading_found:
            title = stripped[2:].strip() or title
            heading_found = True
        source_match = _SOURCE_LINE.match(stripped)
        if source_match:
            url = source_match.group(1)

    return {
        "source": relative_path.as_posix(),
        "title": title,
        "doc_type": relative_path.parts[0] if len(relative_path.parts) > 1 else "legal",
        "url": url,
    }


def _load_cache() -> dict[str, dict[str, Any]]:
    """Đọc mapping document ID từ cache file nếu tồn tại."""
    if CACHE_FILE.exists():
        try:
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("Không thể đọc cache file %s: %s", CACHE_FILE, exc)
    return {}


def _save_cache(cache: dict[str, dict[str, Any]]) -> None:
    """Lưu mapping document ID vào cache file."""
    try:
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Không thể lưu cache file %s: %s", CACHE_FILE, exc)


def _convert_markdown_to_pdf(md_path: Path, pdf_path: Path) -> Path:
    """Chuyển đổi file Markdown sang PDF tạm thời để upload lên PageIndex."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    content = md_path.read_text(encoding="utf-8")

    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Helvetica", size=10)

        # Encode an toàn với font chuẩn latin-1
        safe_text = content.encode("latin-1", "replace").decode("latin-1")
        for line in safe_text.splitlines():
            pdf.multi_cell(w=0, h=5, text=line)

        pdf.output(str(pdf_path))
    except Exception as exc:
        logger.warning("FPDF conversion failed for %s: %s. Using minimal PDF wrapper.", md_path, exc)
        with open(pdf_path, "wb") as f:
            f.write(
                b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
                b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
                b"xref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\n"
                b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF\n"
            )
    return pdf_path


def upload_documents() -> dict[str, dict[str, Any]]:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    api_key = os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY).strip()
    if not api_key:
        logger.info("PAGEINDEX_API_KEY chưa được cấu hình. Bỏ qua upload.")
        return _load_cache()

    if not STANDARDIZED_DIR.exists():
        logger.warning("Thư mục standardized không tồn tại: %s", STANDARDIZED_DIR)
        return {}

    cache = _load_cache()
    TEMP_PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Khởi tạo client nếu SDK khả dụng
    client = None
    try:
        from pageindex import PageIndexClient

        client = PageIndexClient(api_key=api_key)
    except Exception as exc:
        logger.debug("PageIndexClient không khả dụng hoặc lỗi: %s", exc)

    for md_path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_path.name.startswith("."):
            continue
        rel_parts = md_path.relative_to(STANDARDIZED_DIR).parts
        if len(rel_parts) < 2 or rel_parts[0] not in {"legal", "news"}:
            continue

        rel_path = md_path.relative_to(STANDARDIZED_DIR).as_posix()
        if rel_path in cache and cache[rel_path].get("doc_id"):
            continue

        meta = _extract_metadata(md_path)
        pdf_path = TEMP_PDF_DIR / f"{md_path.stem}.pdf"
        _convert_markdown_to_pdf(md_path, pdf_path)

        doc_id = None
        if client is not None:
            try:
                resp = client.submit_document(str(pdf_path))
                if isinstance(resp, dict):
                    doc_id = resp.get("doc_id") or resp.get("id") or resp.get("document_id")
                elif isinstance(resp, str):
                    doc_id = resp
            except Exception as exc:
                logger.warning("Lỗi submit tài liệu %s qua PageIndex SDK: %s", rel_path, exc)

        # Fallback sang REST API nếu SDK không trả về doc_id
        if not doc_id:
            try:
                import requests

                headers = {"Authorization": f"Bearer {api_key}"}
                with open(pdf_path, "rb") as f:
                    files = {"file": (pdf_path.name, f, "application/pdf")}
                    res = requests.post(
                        "https://api.pageindex.ai/v1/documents",
                        headers=headers,
                        files=files,
                        timeout=DEFAULT_TIMEOUT,
                    )
                if res.status_code in (200, 201):
                    data = res.json()
                    doc_id = data.get("doc_id") or data.get("id") or data.get("document_id")
            except Exception as exc:
                logger.warning("Lỗi upload tài liệu %s qua REST API: %s", rel_path, exc)

        if doc_id:
            cache[rel_path] = {
                "doc_id": doc_id,
                "source": meta["source"],
                "title": meta["title"],
                "doc_type": meta["doc_type"],
                "url": meta["url"],
            }
            _save_cache(cache)
            logger.info("Đã upload %s -> doc_id: %s", rel_path, doc_id)

    return cache


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    api_key = os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY).strip()
    if not api_key:
        return []

    cache = _load_cache()
    if not cache:
        return []

    results: list[dict] = []
    seen_ids: set[str] = set()

    try:
        # Thử truy vấn qua SDK hoặc REST API
        client = None
        try:
            from pageindex import PageIndexClient

            client = PageIndexClient(api_key=api_key)
        except Exception:
            pass

        rank = 0
        for rel_path, doc_info in cache.items():
            doc_id = doc_info.get("doc_id")
            if not doc_id:
                continue

            response_data = None
            if client is not None:
                try:
                    if hasattr(client, "chat_completions"):
                        response_data = client.chat_completions(
                            messages=[{"role": "user", "content": query}],
                            doc_id=doc_id,
                        )
                    elif hasattr(client, "chat"):
                        response_data = client.chat(query, doc_id=doc_id)
                    elif hasattr(client, "retrieve"):
                        response_data = client.retrieve(query, doc_id=doc_id)
                except Exception as exc:
                    logger.debug("Lỗi query doc %s qua SDK: %s", doc_id, exc)

            if response_data is None:
                try:
                    import requests

                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    }
                    res = requests.post(
                        "https://api.pageindex.ai/v1/chat/completions",
                        headers=headers,
                        json={
                            "messages": [{"role": "user", "content": query}],
                            "doc_id": doc_id,
                        },
                        timeout=DEFAULT_TIMEOUT,
                    )
                    if res.status_code == 200:
                        response_data = res.json()
                except Exception as exc:
                    logger.debug("Lỗi query doc %s qua REST API: %s", doc_id, exc)

            # Parse content từ response_data
            content = ""
            if isinstance(response_data, dict):
                choices = response_data.get("choices")
                if isinstance(choices, list) and choices:
                    first_choice = choices[0]
                    if isinstance(first_choice, dict):
                        message = first_choice.get("message", {})
                        content = message.get("content", "")
                if not content:
                    content = response_data.get("answer") or response_data.get("content") or ""
            elif isinstance(response_data, str):
                content = response_data.strip()

            if content and isinstance(content, str) and content.strip():
                item_id = f"pi-{doc_id}-{rank}"
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)

                # Gán score giảm dần theo rank
                score = round(1.0 / (1.0 + rank), 4)
                item: dict[str, Any] = {
                    "id": item_id,
                    "content": content.strip(),
                    "score": score,
                    "metadata": {
                        "source": doc_info.get("source", rel_path),
                        "title": doc_info.get("title", Path(rel_path).stem),
                        "doc_type": doc_info.get("doc_type", "legal"),
                        "url": doc_info.get("url"),
                        "chunk_index": rank,
                    },
                    "retrieval_method": "pageindex",
                }
                results.append(item)
                rank += 1

                if len(results) >= top_k:
                    break

    except Exception as exc:
        logger.warning("Lỗi xử lý pageindex_search: %s", exc)
        return []

    # Sort giảm dần theo score và giới hạn top_k
    sorted_results = sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
    return sorted_results


if __name__ == "__main__":
    upload_documents()
