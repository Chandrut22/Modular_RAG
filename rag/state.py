from pydantic import BaseModel, Field
from typing import Literal, List, TypedDict
from langchain_core.documents import Document

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

from typing import List

from pydantic import BaseModel, Field


class EnhancedQueries(BaseModel):
    queries: List[str] = Field(
        description="Search queries: the original ones plus alternative phrasings and documentation-specific terms."
    )


class RAGState(TypedDict, total=False):
    question: str
    plan: QueryPlan
    queries: List[str]
    vector_docs: List[List[Document]]
    keyword_docs: List[List[Document]]
    context: List[Document]
    answer: str
