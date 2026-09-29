from langchain_openai import OpenAIEmbeddings

from .config import OPENAI_API_KEY, EMBEDDING_MODEL


def create_embeddings():
    """Create the OpenAI embedding model."""

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing. "
            "Please add it to the .env file."
        )

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL
    )

    print("Embedding model initialized successfully.")

    return embeddings


if __name__ == "_main_":
    create_embeddings()