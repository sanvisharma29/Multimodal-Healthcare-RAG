import sys
sys.path.append("src")

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_groq_client():
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    except:
        api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key)


def generate_answer(question, retrieved_chunks):
    client = get_groq_client()
    context = "\n\n".join(retrieved_chunks)

    prompt = (
        "You are a helpful medical assistant that explains lab reports in simple language.\n\n"
        "Here is the relevant information from the patient's lab report:\n"
        + context +
        "\n\nNow answer this question in simple language:\n"
        + question +
        "\n\nImportant:\n"
        "- Mention the actual value and whether it is normal, high or low\n"
        "- If a value is at the very bottom or top of the normal range, say it is borderline\n"
        "- If something is abnormal or borderline, explain what it might mean\n"
        "- Never just say normal if the value is at the edge of the range\n"
        "- Always end with: Please consult your doctor for proper medical advice."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    from src.retriever import search_chunks

    question = "What is my Vitamin D level and is it normal?"
    docs, metas = search_chunks(question)
    answer = generate_answer(question, docs)
    print(answer)