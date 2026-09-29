from .loader import load_pdf, split_documents
from .vectorstore import create_vectorstore


def run_ingestion():
    """Load the PDF, split it, and store embeddings in Pinecone."""

    print("Starting document ingestion...")

    documents = load_pdf()

    chunks = split_documents(documents)

    create_vectorstore(chunks)

    print("Ingestion completed successfully!")


if __name__ == "__main__":
    run_ingestion()