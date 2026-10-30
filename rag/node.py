from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException

from rag.llm import make_llm
from rag.fusion import reciprocal_rank_fusion
from rag.retriever import retrieve
from rag.prompt import ANSWER_PROMPT, PLAN_PROMPT, ENHANCE_PROMPT
from rag.state import RAGState, QueryPlan, EnhancedQueries


TOP_K = 4       # chunks retrieved per query, per retriever
TOP_N = 5       # chunks kept after fusion
MAX_QUERIES = 5


llm = make_llm(temperature=0.7)                                          # answer generation
planner = make_llm(temperature=0).with_structured_output(               # routing must be deterministic
    QueryPlan, method="json_mode"
)
enhancer = make_llm(temperature=0).with_structured_output(
    EnhancedQueries, method="json_mode"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def base_queries(plan: QueryPlan) -> List[str]:
    return plan.sub_queries or [plan.rewritten_query]

def queries_for(plan: QueryPlan) -> List[str]:
    return plan.sub_queries or [plan.rewritten_query]


def format_context(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[{i}] ({d.metadata.get('source', '')})\n{d.page_content.strip()}"
        for i, d in enumerate(docs, 1)
    )


def analyze_query(state: RAGState) -> Dict[str, Any]:
    try:
        plan = planner.invoke(PLAN_PROMPT.format(question=state["question"]))
    except OutputParserException as e:
        print(f"[analyze_query] bad planner output, falling back to hybrid: {e}")
        plan = QueryPlan(
            route="hybrid", rewritten_query=state["question"], sub_queries=[]
        )
    return {"plan": plan}

def enhance_query(state: RAGState) -> Dict[str, Any]:
    """Expand the base queries with alternative phrasings / doc terminology."""
    base = base_queries(state["plan"])
    try:
        result = enhancer.invoke(
            ENHANCE_PROMPT.format(
                base_queries="\n".join(f"- {q}" for q in base),
                max_queries=MAX_QUERIES,
            )
        )
        candidates = base + result.queries  # base queries always come first
    except OutputParserException as e:
        print(f"[enhance_query] bad output, using base queries: {e}")
        candidates = base
 
    # dedupe (case-insensitive), keep order, cap the count
    seen, queries = set(), []
    for q in candidates:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            queries.append(q.strip())
    return {"queries": queries[:MAX_QUERIES]}
 


def route_query(state: RAGState) -> List[str]:
    """Conditional edge: returns the node(s) to run next. Two names = parallel."""
    route = state["plan"].route
    if route == "vector":
        return ["vector_retrieve"]
    if route == "keyword":
        return ["keyword_retrieve"]
    return ["vector_retrieve", "keyword_retrieve"]


def vector_retrieve(state: RAGState) -> Dict[str, Any]:
    """Dense embeddings: semantic match. One ranked list per query."""
    return {
        "vector_docs": [
            retrieve(q, mode="dense", k=TOP_K) for q in queries_for(state["plan"])
        ]
    }


def keyword_retrieve(state: RAGState) -> Dict[str, Any]:
    """BM25 sparse vectors: exact-term match. One ranked list per query."""
    return {
        "keyword_docs": [
            retrieve(q, mode="sparse", k=TOP_K) for q in queries_for(state["plan"])
        ]
    }


# ---------------------------------------------------------------------------
# 3. Fusion
# ---------------------------------------------------------------------------
def fuse(state: RAGState) -> Dict[str, Any]:
    ranked_lists = state.get("vector_docs", []) + state.get("keyword_docs", [])
    return {"context": reciprocal_rank_fusion(ranked_lists, top_n=TOP_N)}


# ---------------------------------------------------------------------------
# 4. Generation
# ---------------------------------------------------------------------------
def generate(state: RAGState) -> Dict[str, Any]:
    docs = state.get("context", [])
    if not docs:
        return {"answer": "I could not find anything relevant in the docs."}
    prompt = ANSWER_PROMPT.format(
        context=format_context(docs), question=state["question"]
    )
    return {"answer": llm.invoke(prompt).content}