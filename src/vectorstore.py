import os
from langchain_community.vectorstores import Chroma
from embeddings import create_embeddings

# Chroma DB save ayye folder
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "agentic_ai_docs"

def load_vectorstore():
    """Load existing vectorstore with free embeddings"""
    embeddings = create_embeddings()
    
    # DB already unda leda check
    if not os.path.exists(CHROMA_DB_DIR):
        raise ValueError(f"Vectorstore not found at {CHROMA_DB_DIR}. Please run ingestion.py first")
    
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    return vectorstore

def create_vectorstore_from_docs(docs):
    """Create new vectorstore from documents"""
    embeddings = create_embeddings()
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME
    )
    vectorstore.persist()
    print(f"Vectorstore created at {CHROMA_DB_DIR}")
    return vectorstore