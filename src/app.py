import streamlit as st

from src.graph import build_rag_graph


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Agentic AI RAG Assistant",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# App Title
# --------------------------------------------------

st.title("🤖 Agentic AI RAG Assistant")

st.write(
    "Ask a question about the Agentic AI document."
)


# --------------------------------------------------
# Load RAG Graph
# --------------------------------------------------

@st.cache_resource
def get_rag_graph():
    return build_rag_graph()


# --------------------------------------------------
# User Question
# --------------------------------------------------

question = st.text_input(
    "Enter your question:",
    placeholder="What is Agentic AI?"
)


# --------------------------------------------------
# Ask Button
# --------------------------------------------------

if st.button("Ask"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching the document..."
        ):

            try:

                # Build RAG graph
                graph = get_rag_graph()

                # Run RAG workflow
                result = graph.invoke(
                    {
                        "question": question,
                        "context": [],
                        "answer": "",
                        "score": 0.0
                    }
                )


                # --------------------------------------------------
                # Answer
                # --------------------------------------------------

                st.subheader("Answer")

                st.write(
                    result["answer"]
                )


                # --------------------------------------------------
                # Relevance Score
                # --------------------------------------------------

                st.subheader(
                    "Relevance Score"
                )

                st.write(
                    f"{result['score']:.2f}"
                )


                # --------------------------------------------------
                # Retrieved Context
                # --------------------------------------------------

                st.subheader(
                    "Retrieved Context"
                )

                if result["context"]:

                    for i, chunk in enumerate(
                        result["context"],
                        start=1
                    ):

                        with st.expander(
                            f"Chunk {i}"
                        ):

                            st.write(chunk)

                else:

                    st.write(
                        "No relevant context found."
                    )


            except Exception as e:

                st.error(
                    f"Error: {e}"
                )