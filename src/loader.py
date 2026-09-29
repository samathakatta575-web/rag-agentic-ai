from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf():
    """Load the PDF document."""

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"PDF loaded successfully.")
    print(f"Total pages: {len(documents)}")

    return documents


def split_documents(documents):
    """Split documents into smaller chunks for RAG."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Documents split successfully.")
    print(f"Total chunks: {len(chunks)}")

    return chunks


if __name__ == "_main_":
    documents = load_pdf()
    chunks = split_documents(documents)

    print("\nFirst chunk:")
    print(chunks[0].page_content[:500])