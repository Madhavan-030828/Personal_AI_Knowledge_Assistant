from importlib.metadata import metadata
import re
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from Retrieval.vector_retriever import content_id

#Converts the text into lower case and using regex seperates every word and returns a list of lowercase words.
def _tokenize(text: str):
    """Lightweight tokenizer. Swap for a proper tokenizer (e.g. spaCy) if needed."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """Lexical (keyword) retriever built from the same chunks as the vector store."""

    def __init__(self):
        self.bm25 = None
        self.documents: list[Document] = []
        self.ids: list[str] = []

    def index_from_vectorstore(self, vector_retriever):
        """
        Pulls every stored chunk out of Chroma and builds the BM25 index from
        it, so BM25 and vector search always operate over the same corpus.
        """
        stored = vector_retriever.get_all_documents()

        self.documents = []
        self.ids = []
        for _, text, metadata in stored:
            doc = Document(page_content=text, metadata=metadata or {})
            self.documents.append(doc)
            self.ids.append(content_id(text, metadata))

        tokenized_corpus = [_tokenize(doc.page_content) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 20):
        """Returns a list of (doc_id, Document, score) tuples, best first."""
        if self.bm25 is None:
            raise RuntimeError("Call index_from_vectorstore() before search().")

        scores = self.bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            (self.ids[i], self.documents[i], float(scores[i]))
            for i in top_idx
        ]