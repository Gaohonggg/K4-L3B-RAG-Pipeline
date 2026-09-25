"""Load, chunk, embed, and index the standardized Markdown corpus.

The collection is a snapshot of ``data/standardized``. Document and chunk IDs
are derived from relative paths, so repeated indexing updates existing chunks
and removes chunks that are no longer present in the corpus.
"""

import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from .contracts import validate_document


ROOT_DIR = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = ROOT_DIR / "data" / "standardized"
CHROMA_DIR = ROOT_DIR / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"
EMBEDDING_BATCH_SIZE = 64
CHROMA_BATCH_SIZE = 100
COLLECTION_NAME = "rag_documents"

load_dotenv(ROOT_DIR / ".env")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip()

_SOURCE_LINE = re.compile(r"^\*\*(?:Source|URL):\*\*\s*(\S+)", re.IGNORECASE)


def _embedding_config() -> str:
    """Return the configured model, failing before any paid API call."""
    if EMBEDDING_PROVIDER != "openai":
        raise ValueError("Task 4 requires EMBEDDING_PROVIDER=openai in .env")
    if not EMBEDDING_MODEL:
        raise ValueError("Set EMBEDDING_MODEL in .env")
    return EMBEDDING_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with OpenAI, preserving input order across API batches."""
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding inputs must be non-empty strings")

    model = _embedding_config()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in .env")

    client = OpenAI()
    embeddings: list[list[float]] = []
    expected_dim: int | None = None
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(model=model, input=batch)
        ordered: list[list[float] | None] = [None] * len(batch)
        for item in response.data:
            if not 0 <= item.index < len(batch) or ordered[item.index] is not None:
                raise ValueError("OpenAI returned invalid embedding indices")
            ordered[item.index] = item.embedding
        if any(vector is None or not vector for vector in ordered):
            raise ValueError("OpenAI returned an incomplete embedding batch")
        for vector in ordered:
            if vector is None:
                raise ValueError("OpenAI returned an incomplete embedding batch")
            if expected_dim is None:
                expected_dim = len(vector)
            elif len(vector) != expected_dim:
                raise ValueError("OpenAI returned inconsistent embedding dimensions")
            embeddings.append(vector)
    return embeddings


def get_collection():
    """Open the persistent cosine collection for the configured model."""
    model = _embedding_config()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    expected_metadata = {
        "hnsw:space": "cosine",
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": model,
    }
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=expected_metadata,
    )
    metadata = collection.metadata or {}
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError(
            "Existing Chroma collection uses another embedding configuration; "
            "use a new collection before indexing"
        )
    return collection


def load_documents() -> list[dict]:
    """Load legal/news Markdown with stable IDs and source metadata."""
    documents: list[dict] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        relative_path = path.relative_to(STANDARDIZED_DIR)
        if len(relative_path.parts) < 2 or relative_path.parts[0] not in {"legal", "news"}:
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Empty Markdown document: {relative_path}")

        title = path.stem
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

        document = {
            "id": relative_path.as_posix(),
            "content": content,
            "metadata": {
                "source": relative_path.as_posix(),
                "title": title,
                "doc_type": relative_path.parts[0],
                "url": url,
            },
        }
        validate_document(document)
        documents.append(document)
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into stable, non-empty chunks without losing metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    seen_ids: set[str] = set()
    for document in documents:
        validate_document(document)
        if document["id"] in seen_ids:
            raise ValueError(f"Duplicate document ID: {document['id']}")
        seen_ids.add(document["id"])
        for index, text in enumerate(splitter.split_text(document["content"])):
            if not text.strip():
                continue
            chunk = {
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": {**document["metadata"], "chunk_index": index},
            }
            validate_document(chunk, require_chunk=True)
            chunks.append(chunk)
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Add embeddings in batches without mutating caller-owned chunks."""
    for chunk in chunks:
        validate_document(chunk, require_chunk=True)
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("Embedding count does not match chunk count")
    return [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)]


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert corpus chunks and remove stale IDs from the dedicated collection."""
    if not chunks:
        return
    ids = [chunk["id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk IDs must be unique")
    dimensions = {len(chunk["embedding"]) for chunk in chunks}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("Chunk embeddings must have one non-zero dimension")
    for chunk in chunks:
        validate_document(chunk, require_chunk=True)

    collection = get_collection()
    for start in range(0, len(chunks), CHROMA_BATCH_SIZE):
        batch = chunks[start : start + CHROMA_BATCH_SIZE]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[
                {
                    **chunk["metadata"],
                    "url": chunk["metadata"]["url"] or "",
                }
                for chunk in batch
            ],
        )

    stale_ids = set(collection.get(include=[])["ids"]) - set(ids)
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))


def run_pipeline() -> None:
    """Index the available standardized corpus."""
    documents = load_documents()
    if not documents:
        print(f"No Markdown documents found in {STANDARDIZED_DIR}")
        return
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks from {len(documents)} documents")


if __name__ == "__main__":
    run_pipeline()
