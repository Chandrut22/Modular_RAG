import json
import os
import uuid
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    Modifier,
    SparseVectorParams,
    VectorParams,
)

from rag.chunker import chunk_markdown
from rag.llm import embeddings
from rag.loader import load_langchain_docs

load_dotenv()

INDEX_DIR = Path("index")

QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "langchain_docs")

DENSE_NAME = "dense"
SPARSE_NAME = "sparse"

NAMESPACE = uuid.UUID("b7f1c8e2-3a4d-4e5f-9a6b-1c2d3e4f5a6b")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")


def load_chunks() -> List[Document]:
    docs = load_langchain_docs()
    if not docs:
        raise RuntimeError("No documents were loaded.")

    chunks: List[Document] = []
    for doc in docs:
        chunks.extend(
            chunk_markdown(
                doc.page_content,
                source=doc.metadata["source"],
                max_chars=5000,
                max_code_chars=8000,
            )
        )
    print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks")
    return chunks


def chunk_id(doc: Document) -> str:
    """Deterministic UUID: re-running the build overwrites instead of duplicating."""
    key = f"{doc.metadata['source']}::{doc.metadata['chunk_index']}"
    return str(uuid.uuid5(NAMESPACE, key))


def get_client() -> QdrantClient:
    if not QDRANT_ENDPOINT:
        raise RuntimeError("QDRANT_ENDPOINT is not set")
    if not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_API_KEY is not set")
    return QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)


def get_store(
    client: QdrantClient,
    mode: RetrievalMode = RetrievalMode.HYBRID,
) -> QdrantVectorStore:
    """mode: RetrievalMode.DENSE (vector), SPARSE (BM25) or HYBRID (both, fused)."""
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=mode,
        vector_name=DENSE_NAME,
        sparse_vector_name=SPARSE_NAME,
    )


def build_index_in_qdrant(chunks: List[Document]) -> None:
    dim = len(embeddings.embed_query("dimension probe"))

    client = get_client()
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)  # full rebuild

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={DENSE_NAME: VectorParams(size=dim, distance=Distance.COSINE)},
        sparse_vectors_config={SPARSE_NAME: SparseVectorParams(modifier=Modifier.IDF)},
    )

    store = get_store(client, RetrievalMode.HYBRID)  # HYBRID writes both vectors
    store.add_documents(chunks, ids=[chunk_id(c) for c in chunks], batch_size=64)
    print(
        f"Qdrant: indexed {len(chunks)} chunks into '{COLLECTION}' "
        f"(dense dim={dim} + BM25 sparse)"
    )
    client.close()


def build_index() -> None:
    chunks = load_chunks()
    build_index_in_qdrant(chunks)
    print("Successfully")

if __name__ == "__main__":
    build_index()