"""
citations.py
-------------
Turns retrieved chunks into a numbered source list the LLM can cite
inline (e.g. "[1]", "[2]"), and parses those markers back out of the
generated answer so the UI/API layer can render clickable references.
"""

from __future__ import annotations
from pathlib import Path
import re
from dataclasses import dataclass


@dataclass
class Source:
    """One retrieved chunk, normalized for citation use."""
    index: int          # 1-based, matches the [n] marker used in prompts
    doc_id: str
    title: str | None
    text: str
    metadata: dict | None = None


def build_sources(retrieved_chunks: list[dict]) -> list[Source]:
    """
    Convert raw retriever output into numbered Source objects.

    `retrieved_chunks` is expected to look like what your retrieval
    module returns, e.g.:
        [{"doc_id": "...", "title": "...", "text": "...", "metadata": {...}}, ...]
    """
    sources = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        sources.append(
            Source(
                index=i,
                doc_id=chunk.get("doc_id", f"doc_{i}"),
                title=chunk.get("title"),
                text=chunk["text"],
                metadata=chunk.get("metadata"),
            )
        )
    return sources


def build_sources_from_retriever_results(results: list[tuple]) -> list[Source]:
    sources = []
    for i, (doc_id, doc, score) in enumerate(results, start=1):
        metadata = doc.metadata or {}
        src = metadata.get("source")
        page = metadata.get("page")
        if src and page is not None:
            title = f"{Path(src).name}, p.{page + 1}"
        else:
            title = metadata.get("title") or src

        sources.append(
            Source(
                index=i,
                doc_id=doc_id,
                title=title,
                text=doc.page_content,
                metadata={**metadata, "score": score},
            )
        )
    return sources


def format_context_block(sources: list[Source]) -> str:
    """
    Render sources into the numbered context block injected into the
    prompt, so the model can refer back to them as [1], [2], etc.
    """
    blocks = []
    for s in sources:
        header = f"[{s.index}] {s.title}" if s.title else f"[{s.index}]"
        blocks.append(f"{header}\n{s.text}")
    return "\n\n".join(blocks)


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def extract_cited_indices(answer_text: str) -> list[int]:
    """Return the unique, ordered list of [n] markers used in the answer."""
    seen: list[int] = []
    for match in _CITATION_PATTERN.finditer(answer_text):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def attach_citations(answer_text: str, sources: list[Source]) -> dict:
    """
    Package the final answer with only the sources actually cited,
    ready to serialize as an API response.

    Returns:
        {
            "answer": str,
            "citations": [{"index": int, "doc_id": str, "title": str|None}, ...]
        }
    """
    by_index = {s.index: s for s in sources}
    cited = extract_cited_indices(answer_text)

    citations = [
        {"index": i, "doc_id": by_index[i].doc_id, "title": by_index[i].title}
        for i in cited
        if i in by_index
    ]

    return {"answer": answer_text, "citations": citations}
