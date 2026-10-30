
import os
from typing import Any, Dict, List, Literal, Tuple, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_llm(temperature: float):
    return init_chat_model(
        model="gpt-oss:20b",
        model_provider="openai",
        api_key=os.getenv("API_KEY"),
        base_url="https://ollama.com/v1",
        temperature=temperature,
        max_retries=2,
    )


llm = make_llm(temperature=0.7)          # answer generation
planner_llm = make_llm(temperature=0)    # routing must be deterministic

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

SAMPLE_DOCS = [
    """LangGraph is a library for building stateful, multi-actor applications
    with LLMs. It extends LangChain with the ability to coordinate multiple
    chains (or actors) across multiple steps of computation, in a cyclic
    manner, using a graph-based approach.""",
    """Retrieval-Augmented Generation (RAG) combines a retrieval system with
    a generative model. Instead of relying only on what the model learned
    during training, RAG fetches relevant documents at query time and
    passes them to the model as context before it generates an answer.""",
    """Modular RAG breaks the RAG pipeline into independent, swappable
    stages: query construction, retrieval, post-retrieval processing,
    grading, and generation. Each stage can be modified or replaced
    without rewriting the rest of the pipeline.""",
    """BM25 is a sparse keyword-ranking function. It scores documents by
    term frequency and inverse document frequency, so it excels at exact
    matches such as error codes, product names, and rare terms that
    embeddings can blur together.""",
    """Reciprocal Rank Fusion (RRF) merges several ranked result lists by
    summing 1 / (k + rank) for each document across the lists. Documents
    ranked highly by multiple retrievers rise to the top without needing
    to compare raw scores from different systems.""",
    """FAISS is a library for efficient similarity search over dense
    vectors. A flat index compares the query with every vector (exact but
    slow at scale); HNSW and IVF indexes trade a little accuracy for much
    faster approximate search.""",
]


# ---------------------------------------------------------------------------
# Ingestion (offline): chunk -> embed -> index
# ---------------------------------------------------------------------------
def build_indexes(raw_docs: List[str]) -> Tuple[FAISS, BM25Retriever]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.create_documents(raw_docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    bm25 = BM25Retriever.from_documents(chunks)
    return vectorstore, bm25


# ---------------------------------------------------------------------------
# Query plan (single structured output of analyze_query)
# ---------------------------------------------------------------------------
class QueryPlan(BaseModel):
    route: Literal["vector", "keyword", "hybrid"] = Field(
        description="'keyword' for exact terms/names/acronyms, 'vector' for "
        "conceptual/loosely worded, 'hybrid' if unsure or both apply."
    )
    rewritten_query: str = Field(
        description="Standalone, unambiguous, search-friendly version of the question."
    )
    sub_queries: List[str] = Field(
        default_factory=list,
        description="2-3 self-contained sub-questions if the question is compound, else empty.",
    )


planner = planner_llm.with_structured_output(QueryPlan, method="json_mode")

# NOTE: literal braces are doubled ({{ }}) because this goes through str.format
PLAN_PROMPT = """Analyze this user question for a document-search system.
Return ONLY a JSON object with EXACTLY these three keys and no others:
- "route": one of "vector", "keyword", "hybrid"
    ("keyword" = hinges on exact terms/names/acronyms,
     "vector" = conceptual or loosely worded,
     "hybrid" = unsure or both apply)
- "rewritten_query": a standalone, search-friendly version of the question (string)
- "sub_queries": a list of 2-3 self-contained sub-questions if the question is
  compound (asks several distinct things), otherwise an empty list []

Example output:
{{"route": "keyword", "rewritten_query": "BM25 definition and scoring method", "sub_queries": []}}

Question: {question}"""


# ---------------------------------------------------------------------------
# State — retrievers store one ranked list PER query
# ---------------------------------------------------------------------------
class RAGState(TypedDict, total=False):
    question: str
    plan: QueryPlan
    vector_docs: List[List[Document]]
    keyword_docs: List[List[Document]]
    context: List[Document]
    answer: str


def queries_for(plan: QueryPlan) -> List[str]:
    return plan.sub_queries or [plan.rewritten_query]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def analyze_query(state: RAGState) -> Dict[str, Any]:
    try:
        plan = planner.invoke(PLAN_PROMPT.format(question=state["question"]))
    except OutputParserException as e:
        print(f"[analyze_query] bad planner output, falling back to hybrid: {e}")
        plan = QueryPlan(
            route="hybrid", rewritten_query=state["question"], sub_queries=[]
        )
    return {"plan": plan}  # <-- must be a dict of state keys


def route_query(state: RAGState) -> List[str]:
    """Conditional edge: returns node(s) to run next. Two names = parallel."""
    route = state["plan"].route
    if route == "vector":
        return ["vector_retrieve"]
    if route == "keyword":
        return ["keyword_retrieve"]
    return ["vector_retrieve", "keyword_retrieve"]


def make_vector_node(vectorstore: FAISS, k: int = 4):
    def vector_retrieve(state: RAGState) -> Dict[str, Any]:
        return {
            "vector_docs": [
                vectorstore.similarity_search(q, k=k)
                for q in queries_for(state["plan"])
            ]
        }

    return vector_retrieve


def make_keyword_node(bm25: BM25Retriever, k: int = 4):
    bm25.k = k

    def keyword_retrieve(state: RAGState) -> Dict[str, Any]:
        return {"keyword_docs": [bm25.invoke(q) for q in queries_for(state["plan"])]}

    return keyword_retrieve


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]], k: int = 60, top_n: int = 5
) -> List[Document]:
    scores: Dict[str, float] = {}
    by_key: Dict[str, Document] = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            key = doc.page_content
            by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [by_key[key] for key in ordered[:top_n]]


def fuse(state: RAGState) -> Dict[str, Any]:
    # Only retrievers that actually ran have keys in state.
    ranked_lists = state.get("vector_docs", []) + state.get("keyword_docs", [])
    return {"context": reciprocal_rank_fusion(ranked_lists)}


def generate(state: RAGState) -> Dict[str, Any]:
    context_text = "\n---\n".join(d.page_content.strip() for d in state["context"])
    prompt = f"""Answer the question using only the context below.
If the answer isn't in the context, say you don't know.

Context:
{context_text}

Question: {state['question']}
Answer:"""
    return {"answer": llm.invoke(prompt).content}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------
def build_workflow(vectorstore: FAISS, bm25: BM25Retriever):
    wf = StateGraph(RAGState)
    wf.add_node("analyze_query", analyze_query)
    wf.add_node("vector_retrieve", make_vector_node(vectorstore))
    wf.add_node("keyword_retrieve", make_keyword_node(bm25))
    wf.add_node("fuse", fuse)
    wf.add_node("generate", generate)

    wf.add_edge(START, "analyze_query")
    wf.add_conditional_edges(
        "analyze_query", route_query, ["vector_retrieve", "keyword_retrieve"]
    )
    # Plain edges (not list form): only some retrievers run per query.
    wf.add_edge("vector_retrieve", "fuse")
    wf.add_edge("keyword_retrieve", "fuse")
    wf.add_edge("fuse", "generate")
    wf.add_edge("generate", END)
    return wf.compile()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building indexes...")
    vectorstore, bm25 = build_indexes(SAMPLE_DOCS)
    app = build_workflow(vectorstore, bm25)

    tests = [
        "What does BM25 stand for and score by?",                       # -> keyword
        "how do i merge results coming from different search systems",   # -> vector
        "What is Modular RAG, and what index types does FAISS offer?",   # -> decompose
    ]
    for question in tests:
        result = app.invoke({"question": question})
        plan = result["plan"]
        print("\n" + "=" * 70)
        print("Q:", question)
        print("route:", plan.route)
        print("rewritten:", plan.rewritten_query)
        print("sub_queries:", plan.sub_queries)
        print("context chunks:", len(result["context"]))
        print("\nAnswer:\n", result["answer"])