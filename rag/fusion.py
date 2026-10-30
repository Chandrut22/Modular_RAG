from typing import Dict, List

from langchain_core.documents import Document


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]], k: int = 60, top_n: int = 5
) -> List[Document]:
    """Merge several ranked lists: score = sum of 1 / (k + rank) across lists.
    Chunks ranked high by more than one list rise to the top, and identical
    chunks are merged into one."""
    scores: Dict[str, float] = {}
    by_key: Dict[str, Document] = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            key = doc.page_content
            by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [by_key[key] for key in ordered[:top_n]]