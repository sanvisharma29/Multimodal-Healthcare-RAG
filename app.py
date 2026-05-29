# -*- coding: utf-8 -*-
import sys
sys.path.append("src")

import os
import tempfile
import streamlit as st
from chromadb import PersistentClient
from src.parser import load_pdf, describe_image
from src.chunking import split_by_sections, add_metadata, chunk_image_description
from src.embedding import embed_chunks
from src.retriever import store_chunks, search_chunks
from src.llm import generate_answer

st.set_page_config(
    page_title="MediScan AI",
    page_icon=None,
    layout="wide"
)

client = PersistentClient(path="data/vectorstore")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Source+Sans+3:wght@300;400;500&display=swap');

:root {
    --bg:      #fdf6ee;
    --sidebar: #f5ebe0;
    --card:    #fffaf4;
    --border:  #e0cfc0;
    --brown:   #6b4c35;
    --dark:    #3b2a1e;
    --mid:     #8c6248;
    --muted:   #a08070;
    --text:    #2e1f14;
    --accent:  #c8956a;
    --white:   #ffffff;
}

[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.main .block-container {
    background-color: var(--bg) !important;
    font-family: 'Source Sans 3', sans-serif;
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1200px; }

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: var(--sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: var(--dark) !important;
    font-family: 'Source Sans 3', sans-serif !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border) !important; }
[data-testid="stSidebar"] .stButton > button {
    background-color: var(--card) !important;
    color: var(--dark) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1rem !important;
    font-weight: 400 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: var(--border) !important;
}

.hero {
    background: linear-gradient(135deg, #f0e6d8 0%, #e8d5c0 100%);
    border-radius: 18px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 2rem;
    border: 1px solid var(--border);
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    color: var(--dark);
    margin: 0 0 0.4rem 0;
    font-weight: 600;
}
.hero p {
    color: var(--mid);
    margin: 0;
    font-size: 1rem;
    font-weight: 300;
}

.upload-zone {
    background: var(--card);
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.upload-zone-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--dark);
    margin-bottom: 0.3rem;
}
.upload-zone-sub {
    font-size: 0.85rem;
    color: var(--muted);
}

.info-card {
    background: var(--card);
    border-radius: 14px;
    padding: 1.4rem;
    border: 1px solid var(--border);
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    color: var(--dark);
    margin-bottom: 0.8rem;
    font-weight: 600;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}
.patient-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--dark);
    font-weight: 600;
}
.patient-meta { font-size: 0.82rem; color: var(--muted); margin-top: 0.3rem; }

.abnormal-card {
    background: #fff8f2;
    border: 1px solid #e8c9a8;
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: var(--text);
}

.answer-box {
    background: var(--card);
    border-radius: 14px;
    padding: 1.6rem;
    border: 1px solid var(--border);
    font-size: 0.97rem;
    line-height: 1.85;
    color: var(--text);
    margin-top: 0.8rem;
    box-shadow: 0 2px 10px rgba(107,76,53,0.06);
}
.question-label {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 1.2rem;
    margin-bottom: 0.2rem;
    font-style: italic;
}
.source-chip {
    display: inline-block;
    background: #efe5da;
    color: var(--brown);
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    margin: 0.2rem;
    border: 1px solid var(--border);
}

/* Hide the dark uploader box completely */
[data-testid="stFileUploader"] {
    position: relative !important;
    margin-top: -120px !important;
    opacity: 0 !important;
    height: 120px !important;
    cursor: pointer !important;
}
[data-testid="stFileUploadDropzone"] {
    border: none !important;
    border-radius: 0 0 16px 16px !important;
}
[data-testid="stFileUploadDropzone"] * {
    color: var(--dark) !important;
}
[data-testid="stFileUploadDropzone"] button {
    background-color: var(--dark) !important;
    color: var(--white) !important;
    border-radius: 8px !important;
    border: none !important;
}
[data-testid="stFileUploadDropzone"] button * {
    color: var(--white) !important;
    background-color: transparent !important;
}
[data-testid="stFileUploadDropzone"] svg {
    fill: var(--accent) !important;
}

.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid var(--border) !important;
    padding: 0.7rem 1rem !important;
    font-family: 'Source Sans 3', sans-serif !important;
    background: var(--card) !important;
    color: var(--text) !important;
    font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(200,149,106,0.15) !important;
}
.stTextInput > div > div > input::placeholder { color: var(--muted) !important; }

.stButton > button {
    background: var(--dark) !important;
    color: var(--white) !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.55rem 1.6rem !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 500 !important;
}
.stButton > button:hover { background: var(--brown) !important; }

.stExpander {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
            [data-testid="stFileUploader"] > div {
    border: none !important;
    padding: 0 !important;
}
.stSpinner > div { border-top-color: var(--accent) !important; }
hr { border-color: var(--border) !important; }

</style>
""", unsafe_allow_html=True)


def extract_abnormal_values(chunks):
    abnormals = []
    for chunk in chunks:
        text = chunk.get("text", "")
        for line in text.split("\n"):
            if " H " in line or " L " in line:
                abnormals.append(line.strip()[:80])
    return abnormals[:8]


# Hero
st.markdown("""
<div class="hero">
    <h1>MediScan AI</h1>
    <p>Upload your medical report or scan - ask anything, understand everything.</p>
</div>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.markdown("#### Try asking")
    suggestions = [
        "What is my Vitamin D level?",
        "Are my cholesterol levels normal?",
        "What does my blood sugar indicate?",
        "Summarize all abnormal values",
        "What is my haemoglobin?",
    ]
    for s in suggestions:
        if st.button(s, key=s, use_container_width=True):
            st.session_state.suggested_question = s
    st.markdown("---")
    st.markdown('<p style="font-size:0.75rem;color:#a08070">For informational purposes only. Always consult your doctor.</p>', unsafe_allow_html=True)


# Upload zone
if not st.session_state.get("processed_file"):
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-zone-title">Upload your medical report</div>
        <div class="upload-zone-sub">Supports PDF lab reports, X-rays, MRI scans, CT scans</div>
    </div>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload your report (PDF, JPG, PNG)",
    type=["pdf", "jpg", "jpeg", "png"],
)

if uploaded_file:

    if st.session_state.get("processed_file") != uploaded_file.name:
        with st.spinner("Analysing your report..."):
            try:
                client.delete_collection("user_report")
            except:
                pass
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            ext = uploaded_file.name.split(".")[-1].lower()
            if ext == "pdf":
                text = load_pdf(tmp_path)
                chunks = split_by_sections(text)
                chunks = add_metadata(chunks, "Patient", "Unknown")
                st.session_state.is_image = False
            else:
                description = describe_image(tmp_path)
                chunks = chunk_image_description(description, uploaded_file.name)
                st.session_state.is_image = True
                st.session_state.image_description = description
            chunks = embed_chunks(chunks)
            store_chunks(chunks, collection_name="user_report")
            st.session_state.processed_file = uploaded_file.name
            st.session_state.chunks = chunks
            st.session_state.chat_history = []

    st.markdown("---")
    col1, col2 = st.columns([1.2, 1.8], gap="large")

    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="patient-name">Patient Report</div>
            <div class="patient-meta">""" + uploaded_file.name + """</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("is_image"):
            st.image(uploaded_file, use_container_width=True, caption="Uploaded scan")
            with st.expander("AI Image Analysis"):
                st.write(st.session_state.get("image_description", ""))
        else:
            st.markdown('<div class="info-card"><div class="card-title">Flagged Values</div>', unsafe_allow_html=True)
            abnormals = extract_abnormal_values(st.session_state.get("chunks", []))
            if abnormals:
                for a in abnormals:
                    st.markdown(f'<div class="abnormal-card">{a}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:#4a7c5e;font-size:0.9rem">No obvious abnormal flags detected</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card-title" style="font-family:Playfair Display,serif;font-size:1rem;color:#3b2a1e;font-weight:600;border-bottom:1px solid #e0cfc0;padding-bottom:0.5rem;margin-bottom:1rem">Ask about your report</div>', unsafe_allow_html=True)
        default_q = st.session_state.pop("suggested_question", "")
        question = st.text_input(
            "Your question",
            value=default_q,
            placeholder="e.g. What is my glucose level?",
            label_visibility="collapsed"
        )
        if question:
            with st.spinner("Finding answer..."):
                docs, metas = search_chunks(question, collection_name="user_report")
                answer = generate_answer(question, docs)
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "sources": metas
            })
        for chat in reversed(st.session_state.get("chat_history", [])):
            st.markdown(f'<div class="question-label">"{chat["question"]}"</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{chat["answer"]}</div>', unsafe_allow_html=True)
            sections = list(set([m["section"] for m in chat["sources"]]))
            chips = " ".join([f'<span class="source-chip">{s}</span>' for s in sections])
            st.markdown(f'<div style="margin-top:0.4rem">Sources: {chips}</div>', unsafe_allow_html=True)