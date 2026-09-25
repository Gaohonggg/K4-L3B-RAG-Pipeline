"""Task 3: standardize landing data as UTF-8 Markdown files.

Legal PDF/DOC/DOCX files are converted with MarkItDown. News JSON files
retain their source metadata in a small Markdown header. Output paths are
stable, so running the task again updates existing files without duplicates.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
LANDING_DIR = ROOT_DIR / "data" / "landing"
OUTPUT_DIR = ROOT_DIR / "data" / "standardized"

LEGAL_EXTENSIONS = {".pdf", ".doc", ".docx"}
NEWS_REQUIRED_FIELDS = ("url", "title", "date_crawled", "content_markdown")
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _write_markdown(path: Path, content: str) -> None:
    """Write one normalized, non-empty Markdown document."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError(f"Refusing to create an empty Markdown file: {path.name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{normalized}\n", encoding="utf-8")


def _landing_files(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Landing directory does not exist: {directory}")
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in extensions
        ),
        key=lambda path: path.name.casefold(),
    )


def _ensure_unique_output_stems(paths: list[Path], source_type: str) -> None:
    """Prevent two source files from silently overwriting the same .md file."""
    seen: dict[str, Path] = {}
    for path in paths:
        key = path.stem.casefold()
        if key in seen:
            raise ValueError(
                f"Duplicate {source_type} output name: {seen[key].name!r} and "
                f"{path.name!r} would both become {path.stem}.md"
            )
        seen[key] = path


def convert_legal_docs() -> list[Path]:
    """Convert every legal PDF/DOC/DOCX into ``standardized/legal``."""
    from markitdown import MarkItDown

    source_paths = _landing_files(LANDING_DIR / "legal", LEGAL_EXTENSIONS)
    _ensure_unique_output_stems(source_paths, "legal")

    converter = MarkItDown()
    output_dir = OUTPUT_DIR / "legal"
    written: list[Path] = []
    for source_path in source_paths:
        result = converter.convert(str(source_path))
        # MarkItDown's current result exposes ``markdown``; older releases
        # used ``text_content``. Support both so valid PDF text is retained.
        content = getattr(result, "markdown", None)
        if content is None:
            content = getattr(result, "text_content", None)
        if not isinstance(content, str) or not content.strip():
            warnings.warn(
                f"MarkItDown returned no text for {source_path.name}; skipping it. "
                "The PDF may need OCR.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        output_path = output_dir / f"{source_path.stem}.md"
        _write_markdown(output_path, content)
        written.append(output_path)
    return written


def _read_news_json(path: Path) -> dict[str, str]:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path.name}: {exc.msg}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"News file must contain a JSON object: {path.name}")

    missing = [field for field in NEWS_REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ValueError(f"{path.name} is missing fields: {', '.join(missing)}")

    data: dict[str, str] = {}
    for field in NEWS_REQUIRED_FIELDS:
        value = raw[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path.name}: {field!r} must be a non-empty string")
        data[field] = value.strip()
    return data


def _news_body(data: dict[str, str]) -> str:
    """Keep article/listing content while removing shared site chrome."""
    lines = data["content_markdown"].splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.lstrip().startswith("#### ")
            and line.lstrip("# ").strip() == data["title"]
        ),
        None,
    )
    if start is None:
        start = next(
            (index for index, line in enumerate(lines) if line.lstrip().startswith("#### ")),
            0,
        )
    end = next(
        (
            index
            for index in range(start, len(lines))
            if "Tin liên quan" in lines[index]
            or lines[index].lstrip().startswith("##### Đối tác")
            or lines[index].lstrip().startswith("[](https://twitter.com/share")
        ),
        len(lines),
    )
    body = [
        cleaned.rstrip()
        for line in lines[start:end]
        if (cleaned := _MARKDOWN_IMAGE.sub("", line)).strip()
    ]
    cleaned = "\n".join(body).strip()
    if len(cleaned) < 200:
        raise ValueError(f"News body is too short after cleaning: {data['title']}")
    return cleaned


def _news_markdown(data: dict[str, str]) -> str:
    return (
        f"# {data['title']}\n\n"
        f"**Source:** {data['url']}\n\n"
        f"**Crawled:** {data['date_crawled']}\n\n"
        "---\n\n"
        f"{_news_body(data)}"
    )


def convert_news_articles() -> list[Path]:
    """Convert news JSON files while preserving their metadata and content."""
    source_paths = _landing_files(LANDING_DIR / "news", {".json"})
    _ensure_unique_output_stems(source_paths, "news")

    output_dir = OUTPUT_DIR / "news"
    written: list[Path] = []
    for source_path in source_paths:
        output_path = output_dir / f"{source_path.stem}.md"
        _write_markdown(output_path, _news_markdown(_read_news_json(source_path)))
        written.append(output_path)
    return written


def convert_all() -> tuple[list[Path], list[Path]]:
    """Convert all supported landing data and return the generated paths."""
    legal_outputs = convert_legal_docs()
    news_outputs = convert_news_articles()
    print(
        f"Saved {len(legal_outputs)} legal and {len(news_outputs)} news "
        f"Markdown files to: {OUTPUT_DIR}"
    )
    return legal_outputs, news_outputs


if __name__ == "__main__":
    convert_all()
