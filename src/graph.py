from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from .retriever import create_retriever
from .config import OPENAI_API_KEY, LLM_MODEL


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


def build_rag_graph():
    """Build the LangGraph RAG workflow."""

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing. "
            "Please check your .env file."
        )

    retriever = create_retriever()

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL,
        temperature=0
    )

    def retrieve_node(state: AgentState):
        """Retrieve relevant chunks from the vector database."""

        documents = retriever.invoke(state["question"])

        context = [
            document.page_content
            for document in documents
        ]

        return {
            "context": context
        }

    def generate_node(state: AgentState):
        """Generate an answer using only retrieved context."""

        context_text = "\n\n---\n\n".join(
            state["context"]
        )

        prompt = f"""
You are a strict document-based AI assistant.

You MUST answer the user's question using ONLY the
information provided in the CONTEXT below.

Do not use outside knowledge.
Do not guess.
Do not invent facts.

If the answer is not supported by the context, respond exactly:

I cannot answer based on the provided document.

CONTEXT:
{context_text}

QUESTION:
{state["question"]}
"""

        response = llm.invoke(prompt)

        # Basic retrieval-based confidence score.
        # This is a heuristic, not a probability.
        if len(state["context"]) >= 3:
            score = 0.95
        elif len(state["context"]) > 0:
            score = 0.80
        else:
            score = 0.0

        return {
            "answer": response.content,
            "score": score
        }

    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()