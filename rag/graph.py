from langgraph.graph import END, START, StateGraph

from rag.node import (
    analyze_query,
    fuse,
    generate,
    keyword_retrieve,
    route_query,
    vector_retrieve,
    enhance_query
)
from rag.state import RAGState


def build_graph():
    wf = StateGraph(RAGState)

    wf.add_node("analyze_query", analyze_query)
    wf.add_node("enhance_query", enhance_query)
    wf.add_node("vector_retrieve", vector_retrieve)
    wf.add_node("keyword_retrieve", keyword_retrieve)
    wf.add_node("fuse", fuse)
    wf.add_node("generate", generate)

    wf.add_edge(START, "analyze_query")
    wf.add_edge("analyze_query","enhance_query")
    wf.add_conditional_edges(
        "enhance_query", route_query, ["vector_retrieve", "keyword_retrieve"]
    )
    wf.add_edge("vector_retrieve", "fuse")
    wf.add_edge("keyword_retrieve", "fuse")
    wf.add_edge("fuse", "generate")
    wf.add_edge("generate", END)

    return wf.compile()