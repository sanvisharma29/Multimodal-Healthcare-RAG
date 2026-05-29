import sys
sys.path.append("src")
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
api_key = st.secrets.get("COHERE_API_KEY") or os.getenv("COHERE_API_KEY")
co = cohere.Client(api_key)

def embed_chunks(chunks):
    """
    Takes a list of chunks from chunker.py
    and adds an embedding (vector) to each one.
    """
    # Extract just the text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    # Send to Cohere and get vectors back
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"
    )

    # Add the vector back into each chunk
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = response.embeddings[i]

    return chunks


# Test it
if __name__ == "__main__":
    from src.parser import load_pdf
    from src.chunking import split_by_sections, add_metadata

    # Load and chunk the PDF
    text = load_pdf(r"data\sample_reports\sample_report.pdf")
    chunks = split_by_sections(text)
    chunks = add_metadata(chunks, "Lyubochka Svetka", "20-Feb-2023")

    print(f"Embedding {len(chunks)} chunks...")

    # Embed them
    chunks = embed_chunks(chunks)

    print(f"Done!")
    print(f"Sample embedding (first 5 numbers): {chunks[0]['embedding'][:5]}")
    print(f"Embedding size: {len(chunks[0]['embedding'])}")