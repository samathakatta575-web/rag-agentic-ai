from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from .embeddings import create_embeddings
from .config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME
)


# OpenAI text-embedding-3-small produces 1536-dimensional vectors
INDEX_DIMENSION = 1536
INDEX_METRIC = "cosine"


def get_pinecone_index():
    """Connect to Pinecone and create the index if needed."""

    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY is missing. "
            "Please add it to the .env file."
        )

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [
        index["name"]
        for index in pc.list_indexes()
    ]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print("Creating Pinecone index...")

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=INDEX_DIMENSION,
            metric=INDEX_METRIC,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print("Pinecone index created successfully.")

    else:
        print("Pinecone index already exists.")

    return pc.Index(PINECONE_INDEX_NAME)


def create_vectorstore(chunks):
    """Create Pinecone vector store from document chunks."""

    embeddings = create_embeddings()

    get_pinecone_index()

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )

    print("Documents successfully stored in Pinecone.")

    return vectorstore


def load_vectorstore():
    """Load the existing Pinecone vector store."""

    embeddings = create_embeddings()

    get_pinecone_index()

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    print("Pinecone vector store loaded successfully.")
    return vectorstore