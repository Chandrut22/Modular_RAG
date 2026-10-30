from typing import List

from langchain_core.documents import Document

from rag.graph import build_graph


def format_sources(docs: List[Document]) -> str:
    return "\n".join(
        f"[{i}] {d.metadata.get('section', '')} - {d.metadata.get('source', '')}"
        for i, d in enumerate(docs, 1)
    )


def main() -> None:
    rag = build_graph()

    with open("image.png", "wb") as f:
        f.write(rag.get_graph().draw_mermaid_png())

    print("Modular RAG ready. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break

        result = rag.invoke({"question": question})
        plan = result["plan"]

        print(f"\n[route] {plan.route}")
        print(f"[rewritten] {plan.rewritten_query}")
        if plan.sub_queries:
            print(f"[sub-queries] {plan.sub_queries}")
        print(f"[search queries] {result['queries']}")
        print(f"\n{result['answer']}\n")
        print("Sources:\n" + format_sources(result.get("context", [])) + "\n")


if __name__ == "__main__":
    main()