"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
Implementation trả đoạn nguồn gốc, không đưa chat answer vào context.
"""

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from .contracts import validate_search_results


ROOT_DIR = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = ROOT_DIR / "data" / "standardized"
LANDING_LEGAL_DIR = ROOT_DIR / "data" / "landing" / "legal"
CACHE_FILE = ROOT_DIR / "pageindex_doc_ids.json"
TEMP_PDF_DIR = ROOT_DIR / "data" / "_tmp_pdf"
API_BASE = "https://api.pageindex.ai"
REQUEST_TIMEOUT = 10
SEARCH_TIMEOUT = 30
POLL_INTERVAL = 1

load_dotenv(ROOT_DIR / ".env")
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
_SOURCE_LINE = re.compile(r"^\*\*(?:Source|URL):\*\*\s*(\S+)", re.IGNORECASE)
_PAGE_CITATION = re.compile(r"<doc=[^;>]+;page=(\d+)(?:;[^>]*)?>")
logger = logging.getLogger(__name__)


def _extract_metadata(md_path: Path) -> dict[str, Any]:
    """Trích xuất metadata từ file Markdown chuẩn hóa."""
    relative_path = md_path.relative_to(STANDARDIZED_DIR)
    content = md_path.read_text(encoding="utf-8")
    title = md_path.stem
    heading_found = False
    url: str | None = None
    for line in content.splitlines()[:30]:
        stripped = line.strip()
        if stripped.startswith("# ") and not heading_found:
            title = stripped[2:].strip() or title
            heading_found = True
        match = _SOURCE_LINE.match(stripped)
        if match:
            url = match.group(1)
    return {
        "source": relative_path.as_posix(),
        "title": title,
        "doc_type": relative_path.parts[0],
        "url": url,
    }


def _load_cache() -> dict[str, dict[str, Any]]:
    """Đọc mapping document ID từ cache file nếu tồn tại."""
    if not CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if isinstance(value, dict)}
    except (OSError, ValueError) as exc:
        logger.warning("Cannot read PageIndex cache: %s", exc)
    return {}


def _save_cache(cache: dict[str, dict[str, Any]]) -> None:
    """Lưu mapping document ID vào cache file."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(CACHE_FILE)


def _unicode_font() -> Path:
    configured = os.getenv("PAGEINDEX_PDF_FONT", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise RuntimeError("No Unicode TTF font found; set PAGEINDEX_PDF_FONT")


def _convert_markdown_to_pdf(md_path: Path, pdf_path: Path) -> Path:
    """Chuyển Markdown sang PDF Unicode hợp lệ để upload PageIndex."""
    from fpdf import FPDF

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("Unicode", fname=str(_unicode_font()))
    pdf.add_page()
    pdf.set_font("Unicode", size=10)
    for line in md_path.read_text(encoding="utf-8").splitlines():
        pdf.multi_cell(w=0, h=5, text=line or " ", new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(pdf_path))
    return pdf_path


def _upload_path(md_path: Path) -> Path:
    relative = md_path.relative_to(STANDARDIZED_DIR)
    if relative.parts[0] == "legal":
        original_pdf = LANDING_LEGAL_DIR / f"{md_path.stem}.pdf"
        if original_pdf.is_file():
            return original_pdf
    return _convert_markdown_to_pdf(md_path, TEMP_PDF_DIR / relative.with_suffix(".pdf"))


def upload_documents() -> dict[str, dict[str, Any]]:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    api_key = os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY).strip()
    cache = _load_cache()
    if not api_key or not STANDARDIZED_DIR.exists():
        return cache

    for md_path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        relative = md_path.relative_to(STANDARDIZED_DIR)
        if len(relative.parts) < 2 or relative.parts[0] not in {"legal", "news"}:
            continue
        try:
            upload_path = _upload_path(md_path)
            # Generated PDFs may contain creation metadata, so hash the stable
            # Markdown input and, for original legal PDFs, the original bytes.
            source_bytes = md_path.read_bytes()
            if upload_path.is_relative_to(LANDING_LEGAL_DIR):
                source_bytes += upload_path.read_bytes()
            digest = hashlib.sha256(source_bytes).hexdigest()
            cache_key = relative.as_posix()
            cached = cache.get(cache_key, {})
            if cached.get("doc_id") and cached.get("sha256") == digest:
                continue
            with upload_path.open("rb") as file:
                response = requests.post(
                    f"{API_BASE}/doc/",
                    headers={"api_key": api_key},
                    files={"file": (upload_path.name, file, "application/pdf")},
                    data={"if_retrieval": "true"},
                    timeout=REQUEST_TIMEOUT,
                )
            response.raise_for_status()
            doc_id = response.json().get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("PageIndex upload did not return a doc_id")
            cache[cache_key] = {
                "doc_id": doc_id,
                "sha256": digest,
                **_extract_metadata(md_path),
            }
            _save_cache(cache)
        except Exception as exc:
            logger.warning("PageIndex upload failed for %s: %s", relative, exc)
    return cache


def _retrieved_passages(response: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract original passages from retrieved nodes, excluding chat answers."""
    nodes = response.get("retrieved_nodes")
    if not isinstance(nodes, list):
        result = response.get("result")
        nodes = result.get("retrieved_nodes") if isinstance(result, dict) else result
    if not isinstance(nodes, list):
        return []

    passages: list[tuple[str, str]] = []
    for node_index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id") or node_index)
        content = node.get("content")
        pieces = content if isinstance(content, list) else [node]
        for piece_index, piece in enumerate(pieces):
            if not isinstance(piece, dict):
                continue
            text = piece.get("relevant_content") or piece.get("text")
            if isinstance(text, str) and text.strip():
                page = piece.get("page_index", node.get("page_index", piece_index))
                passages.append((f"{node_id}:{page}:{piece_index}", text.strip()))
    return passages


def _retrieval_passages(
    doc_id: str, query: str, headers: dict[str, str], deadline: float
) -> list[tuple[str, str]]:
    """Use the retrieval endpoint exposed by the pinned PageIndex SDK."""
    remaining = max(0.1, min(REQUEST_TIMEOUT, deadline - time.monotonic()))
    submitted = requests.post(
        f"{API_BASE}/retrieval/",
        headers=headers,
        json={"doc_id": doc_id, "query": query, "thinking": False},
        timeout=remaining,
    )
    submitted.raise_for_status()
    retrieval_id = submitted.json().get("retrieval_id")
    if not isinstance(retrieval_id, str) or not retrieval_id:
        return []

    while time.monotonic() < deadline:
        remaining = max(0.1, min(REQUEST_TIMEOUT, deadline - time.monotonic()))
        response = requests.get(
            f"{API_BASE}/retrieval/{retrieval_id}/",
            headers=headers,
            timeout=remaining,
        )
        response.raise_for_status()
        payload = response.json()
        status = str(payload.get("status", "")).lower()
        if status in {"failed", "error"}:
            return []
        if status == "completed" or "retrieved_nodes" in payload:
            return _retrieved_passages(payload)
        time.sleep(min(POLL_INTERVAL, max(0, deadline - time.monotonic())))
    return []


def _chat_page_passages(
    doc_id: str, query: str, headers: dict[str, str], deadline: float
) -> list[tuple[str, str]]:
    """Resolve current Chat API citations to original OCR page text."""
    if time.monotonic() >= deadline:
        return []
    remaining = max(0.1, min(REQUEST_TIMEOUT, deadline - time.monotonic()))
    response = requests.post(
        f"{API_BASE}/chat/completions",
        headers=headers,
        json={
            "doc_id": doc_id,
            "messages": [{"role": "user", "content": query}],
            "enable_citations": True,
            "stream": False,
        },
        timeout=remaining,
    )
    response.raise_for_status()
    payload = response.json()

    pages: set[int] = set()
    for citation in payload.get("citations", []):
        if isinstance(citation, dict):
            page = citation.get("page", citation.get("page_index"))
            if isinstance(page, int) and page > 0:
                pages.add(page)
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        answer = message.get("content", "") if isinstance(message, dict) else ""
        if isinstance(answer, str):
            pages.update(int(value) for value in _PAGE_CITATION.findall(answer))
    if not pages or time.monotonic() >= deadline:
        return []

    remaining = max(0.1, min(REQUEST_TIMEOUT, deadline - time.monotonic()))
    ocr = requests.get(
        f"{API_BASE}/doc/{doc_id}/",
        headers=headers,
        params={"type": "ocr", "format": "page"},
        timeout=remaining,
    )
    ocr.raise_for_status()
    page_data = ocr.json().get("result")
    if not isinstance(page_data, list):
        return []
    passages: list[tuple[str, str]] = []
    for page in page_data:
        if not isinstance(page, dict) or page.get("page_index") not in pages:
            continue
        text = page.get("markdown")
        if isinstance(text, str) and text.strip():
            passages.append((f"page:{page['page_index']}", text.strip()))
    return passages


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult trong thời gian chờ giới hạn."""
    api_key = os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY).strip()
    if not api_key or not isinstance(query, str) or not query.strip() or top_k < 1:
        return []
    cache = _load_cache()
    if not cache:
        return []

    deadline = time.monotonic() + SEARCH_TIMEOUT
    results: list[dict] = []
    seen_ids: set[str] = set()
    headers = {"api_key": api_key}

    for source, info in sorted(cache.items()):
        doc_id = info.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id or time.monotonic() >= deadline:
            continue
        passages: list[tuple[str, str]] = []
        try:
            passages = _retrieval_passages(doc_id, query, headers, deadline)
        except Exception as exc:
            logger.info("PageIndex retrieval API unavailable for %s: %s", source, exc)
        if not passages and time.monotonic() < deadline:
            try:
                passages = _chat_page_passages(doc_id, query, headers, deadline)
            except Exception as exc:
                logger.warning("PageIndex citation lookup failed for %s: %s", source, exc)

        for passage_id, content in passages:
            item_id = f"pageindex::{doc_id}::{passage_id}"
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            rank = len(results)
            results.append({
                "id": item_id,
                "content": content,
                "score": 1.0 / (rank + 1),
                "metadata": {
                    "source": info.get("source") or source,
                    "title": info.get("title") or Path(source).stem,
                    "doc_type": info.get("doc_type") or source.split("/")[0],
                    "url": info.get("url"),
                    "chunk_index": rank,
                },
                "retrieval_method": "pageindex",
            })
            if len(results) >= top_k:
                break
        if len(results) >= top_k:
            break

    validate_search_results(results, top_k=top_k, expected_method="pageindex")
    return results


if __name__ == "__main__":
    upload_documents()
