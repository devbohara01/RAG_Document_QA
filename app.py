import warnings
warnings.filterwarnings("ignore")

from src.document_loader import load_documents
from src.embedding_store import create_vector_store
from src.retriever import retrieve_documents
from src.llm import get_llm

# Load document
chunks = load_documents("data")

# Create vector store
vector_store = create_vector_store(chunks)

# Load Gemini
llm = get_llm()

print("=" * 60)
print("📚 DOCUMENT QUESTION ANSWERING SYSTEM (RAG)")
print("=" * 60)
print("Type 'exit' to quit.")
print("=" * 60)

while True:

    query = input("\nAsk a Question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    # Retrieve relevant documents with similarity score
    docs = retrieve_documents(vector_store, query)

    # Create context
    context = "\n\n".join([doc.page_content for doc, score in docs])

    # Prompt
    prompt = f"""
You are an intelligent Document Question Answering Assistant.

Rules:
1. Answer ONLY using the provided context.
2. If the answer is not available, say:
   "I couldn't find that information in the provided document."
3. Keep the answer concise and accurate.
4. Do not make up information.

Context:
{context}

Question:
{query}

Answer:
"""

    # Generate Answer
    response = llm.invoke(prompt)

    print("\n" + "=" * 60)
    print("🤖 ANSWER")
    print("=" * 60)
    print(response.content)

    print("\n" + "=" * 60)
    print("📚 SOURCES")
    print("=" * 60)

    for i, (doc, score) in enumerate(docs, start=1):

        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "Unknown")

        print(f"{i}. File : {source}")
        print(f"   Page : {page + 1}")
        print(f"   Distance : {score:.4f}")