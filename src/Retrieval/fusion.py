class ReciprocalRankFusion:
    """
    Merges ranked result lists from different retrievers.
    Uses rank position rather than raw scores, which avoids having to
    normalize BM25 scores and cosine distances onto the same scale.
    """

    def __init__(self, k: int = 60):
        # k is a smoothing constant from the original RRF paper (Cormack et al., 2009)
        self.k = k

    def fuse(self, ranked_lists):
        """
        Each ranked_list is [(doc_id, Document, score), ...], best first.
        Returns [(doc_id, Document, fused_score), ...] sorted best first.
        """
        fused_scores = {}
        doc_lookup = {}

        for ranked_list in ranked_lists:
            for rank, (doc_id, doc, _) in enumerate(ranked_list):
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (self.k + rank + 1)
                doc_lookup.setdefault(doc_id, doc)

        ranked_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, doc_lookup[doc_id], score) for doc_id, score in ranked_ids]
