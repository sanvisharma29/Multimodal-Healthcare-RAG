import sys
sys.path.append("src")

import os
import cohere
import chromadb
from dotenv import load_dotenv

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))
client = chromadb.PersistentClient(path="data/vectorstore")


def store_chunks(chunks, collection_name="healthcare"):
    """
    Saves all chunks with their embeddings into ChromaDB.
    """
    collection = client.get_or_create_collection(name=collection_name)

    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[chunk["embedding"]],
            documents=[chunk["text"]],
            metadatas=[{
                "section": chunk["section"],
                "patient_name": chunk["patient_name"],
                "report_date": chunk["report_date"]
            }]
        )

    print(f"Stored {len(chunks)} chunks in ChromaDB")


def search_chunks(query, collection_name="healthcare", n_results=5):
    """
    Takes a user question, embeds it, searches ChromaDB.
    Returns the most relevant chunks.
    """
    # Embed the user question
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    )
    query_embedding = response.embeddings[0]

    # Search ChromaDB
    collection = client.get_or_create_collection(name=collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results["documents"][0], results["metadatas"][0]


# Test it
if __name__ == "__main__":
    query = "What is my Vitamin D level?"

    print(f"Question: {query}")
    print("Searching...")

    docs, metas = search_chunks(query)

    print(f"\nTop {len(docs)} relevant chunks found:\n")
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        print(f"Result {i+1} — Section: {meta['section']}")
        print(f"{doc[:200]}")
        print("-" * 40)