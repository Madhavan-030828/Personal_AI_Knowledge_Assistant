from Retrieval.vector_retriever import VectorRetriever
from Retrieval.bm25_retriever import BM25Retriever
from Retrieval.fusion import ReciprocalRankFusion
from Retrieval.reranker import Reranker


class HybridRetriever:
    """
    Combines BM25 (lexical) and Chroma vector search (semantic) via
    Reciprocal Rank Fusion, then reorders the fused shortlist with a
    cross-encoder reranker for the final result set.
    """

    def __init__(
        self,
        persist_directory: str,
        embedding_model,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rrf_k: int = 60,
    ):
        self.vector_retriever = VectorRetriever(persist_directory, embedding_model)
        self.bm25_retriever = BM25Retriever()
        self.bm25_retriever.index_from_vectorstore(self.vector_retriever)

        self.fusion = ReciprocalRankFusion(k=rrf_k)
        self.reranker = Reranker(reranker_model_name)

    def retrieve(
        self,
        query: str,
        bm25_top_k: int = 20,
        vector_top_k: int = 20,
        fused_top_k: int = 15,
        final_top_n: int = 5,
    ):
        bm25_results = self.bm25_retriever.search(query, k=bm25_top_k)
        vector_results = self.vector_retriever.search(query, k=vector_top_k)

        fused = self.fusion.fuse([bm25_results, vector_results])
        candidates = fused[:fused_top_k]

        return self.reranker.rerank(query, candidates, top_n=final_top_n)
