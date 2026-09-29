sample_queries = [
    "What is Agentic AI?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "What are the benefits of Agentic AI?",
    "Who won the 2022 FIFA World Cup?"
]

print("Sample RAG Test Queries")
print("=" * 40)

for number, query in enumerate(sample_queries, start=1):
    print(f"{number}. {query}")
