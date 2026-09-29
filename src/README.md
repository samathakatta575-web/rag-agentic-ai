# Agentic AI RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot built using Python, LangGraph, LangChain, OpenAI embeddings, Pinecone, and Streamlit.

The application retrieves relevant information from an Agentic AI eBook and generates answers using the retrieved document context.

---

## Project Overview

This project implements a document-based RAG system for answering questions from an Agentic AI eBook.

The system:

1. Loads the Agentic AI PDF.
2. Splits the document into smaller chunks.
3. Generates embeddings for the chunks.
4. Stores the embeddings in Pinecone.
5. Retrieves relevant chunks for a user question.
6. Uses a LangGraph workflow to generate a grounded answer.
7. Displays the answer, relevance score, and retrieved context through a Streamlit UI.

---

## Tech Stack

- Python 3.10+
- LangChain
- LangGraph
- OpenAI
- Pinecone
- PyPDF
- Streamlit
- python-dotenv

---

## Project Structure

```text
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── _init_.py
│   ├── app.py
│   ├── config.py
│   ├── embeddings.py
│   ├── graph.py
│   ├── ingestion.py
│   ├── loader.py
│   ├── retriever.py
│   ├── vectorstore.py
│   └── tests_sample_queries.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md