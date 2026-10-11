from sentence_transformers import CrossEncoder


class Reranker:
    """
    Reranks candidate chunks against the query with a cross-encoder.
    Cross-encoders jointly encode (query, chunk) pairs, which is far more
    precise than comparing independently-computed embeddings, but too
    expensive to run over the full corpus, so it only scores the shortlist.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates, top_n: int = 5):
        """
        candidates: list of (doc_id, Document, score) tuples from fusion.
        Returns [(doc_id, Document, rerank_score), ...] sorted best first, truncated to top_n.
        """
        pairs = [[query, doc.page_content] for _, doc, _ in candidates]
        scores = self.model.predict(pairs)

        reranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [(doc_id, doc, float(score)) for (doc_id, doc, _), score in reranked[:top_n]]