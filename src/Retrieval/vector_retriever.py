import hashlib
from annotated_types import doc
from langchain_chroma import Chroma

def content_id(text: str, metadata: dict | None = None) -> str:
    m = metadata or {}
    key = f"{m.get('source', '')}|{m.get('page', '')}|{text}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]


class VectorRetriever:
    """Wraps the persisted Chroma vector store for semantic search."""

    def __init__(self, persist_directory: str, embedding_model):
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_metadata={
                "hnsw:space": "cosine"
            }
        )

    def search(self, query: str, k: int = 20):
        """
        Returns a list of (doc_id, Document, score) tuples, best first.
        similarity_search_with_score returns distance, so lower = more similar
        for cosine space; we keep the raw score for transparency.
        """
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [
            (self._doc_id(doc), doc, score)
            for doc, score in results
        ]

    def get_all_documents(self):
        """
        Fetches every stored chunk with its id and metadata — used to build
        the BM25 corpus so both retrievers reference the same chunk ids.
        """
        stored = self.vectorstore.get(include=["documents", "metadatas"])
        return list(zip(stored["ids"], stored["documents"], stored["metadatas"]))

    @staticmethod
    def _doc_id(doc):
        return content_id(doc.page_content, doc.metadata)
