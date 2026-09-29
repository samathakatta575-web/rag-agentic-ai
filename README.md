# Agentic AI - RAG Chatbot

**Live Demo:** [Add your Streamlit link here]

## Tech Stack
- **Vector DB:** Pinecone (384 dim, cosine)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (FREE - No OpenAI billing)
- **LLM:** Groq `openai/gpt-oss-20b` (FREE)
- **Framework:** LangGraph + Streamlit

## How it Works
1. `ingestion.py` - PDF -> chunks -> HuggingFace embeddings -> Pinecone
2. `graph.py` - LangGraph: retrieve (top 3) -> generate
3. `app.py` - Streamlit UI shows Answer + Confidence Score + Retrieved Context

## Run Locally
```bash
pip install -r requirements.txt
streamlit run src/app.py
