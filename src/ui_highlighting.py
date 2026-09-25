"""Safe HTML helpers for highlighting cited evidence in the Streamlit UI."""

from __future__ import annotations

import html
import re


_CITATION = re.compile(r"\[S(\d+)\]")
_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = {
    "bằng",
    "cho",
    "các",
    "của",
    "trong",
    "không",
    "một",
    "những",
    "theo",
    "trên",
    "vào",
    "with",
    "from",
    "that",
    "this",
}


def cited_source_numbers(answer: str) -> set[int]:
    """Return one-based source numbers referenced by an answer."""
    return {int(index) for index in _CITATION.findall(answer)}


def _highlight_terms(answer: str, query: str) -> list[str]:
    text = _CITATION.sub("", f"{answer} {query}")
    terms = {
        token.casefold()
        for token in _TOKEN.findall(text)
        if len(token) >= 4 and token.casefold() not in _STOPWORDS
    }
    return sorted(terms, key=lambda value: (-len(value), value))


def highlight_evidence(content: str, answer: str, query: str = "") -> str:
    """Escape source text and mark answer/query terms without allowing HTML injection."""
    terms = _highlight_terms(answer, query)
    if not terms:
        return html.escape(content).replace("\n", "<br>")

    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(term) for term in terms) + r")(?!\w)",
        re.IGNORECASE,
    )
    pieces: list[str] = []
    cursor = 0
    for match in pattern.finditer(content):
        pieces.append(html.escape(content[cursor : match.start()]))
        pieces.append(f"<mark>{html.escape(match.group(0))}</mark>")
        cursor = match.end()
    pieces.append(html.escape(content[cursor:]))
    return "".join(pieces).replace("\n", "<br>")
