from vectorstore import load_vectorstore


def create_retriever():
    """Create a retriever from the vector database."""

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    print("Retriever created successfully.")

    return retriever