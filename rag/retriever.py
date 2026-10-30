
from typing import List, Literal

from langchain_core.documents import Document
from langchain_qdrant import RetrievalMode

from rag.build_index import get_client, get_store

Mode = Literal["dense", "sparse", "hybrid"]

_MODES = {
    "dense": RetrievalMode.DENSE,
    "sparse": RetrievalMode.SPARSE,
    "hybrid": RetrievalMode.HYBRID,
}

_client = None
_stores = {}


def _get_store(mode: Mode):
    global _client
    if _client is None:
        _client = get_client()
    if mode not in _stores:
        _stores[mode] = get_store(_client, _MODES[mode])
    return _stores[mode]


def retrieve(query: str, mode: Mode = "hybrid", k: int = 5) -> List[Document]:
    return _get_store(mode).similarity_search(query, k=k)


def dedupe(docs: List[Document]) -> List[Document]:
    seen, out = set(), []
    for d in docs:
        key = d.page_content.strip()
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out