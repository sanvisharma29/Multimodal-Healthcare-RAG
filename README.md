# MediScan AI - Multimodal Healthcare RAG

A RAG-based (Retrieval Augmented Generation) medical report assistant that allows users to upload lab reports, X-rays, MRI scans and ask questions about them in plain language.

---

## What it does

Most people receive medical reports full of technical terms and numbers they don't understand. MediScan AI lets you upload any medical document or scan and ask questions like:

- "What is my Vitamin D level?"
- "Is my cholesterol normal?"
- "What does my blood sugar indicate?"
- "Summarize all my abnormal values"

And get back clear, plain-language answers instantly.

---

## Features

- Upload PDF lab reports and get instant answers
- Upload medical images (X-ray, MRI, CT scan) — vision AI analyzes them
- Automatically flags abnormal values from your report
- Chat history — ask multiple questions in one session
- Switches reports automatically when a new file is uploaded

---

## Tech Stack

| Component | Technology |
|---|---|
| PDF Parsing | pdfplumber |
| Medical Image Analysis | Groq Vision LLM (LLaMA 4 Scout) |
| Text Embeddings | Cohere (embed-english-v3.0) |
| Vector Database | ChromaDB |
| LLM Answer Generation | Groq (LLaMA 3.3 70B) |
| Frontend | Streamlit |
| Language | Python |

---

## Project Structure

```
healthcare-rag/
├── src/
│   ├── parser.py        # PDF and image parsing
│   ├── chunker.py       # Section-aware text chunking
│   ├── embedding.py     # Cohere embeddings
│   ├── retriever.py     # ChromaDB storage and search
│   └── llm.py           # Groq LLM answer generation
├── data/
│   ├── uploads/         # Temporary uploaded files
│   └── sample_reports/  # Test reports
├── notebook/            # Jupyter notebooks
├── app.py               # Streamlit UI
├── requirements.txt
└── .env                 
```

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/sanvisharma29/Multimodal-Healthcare-RAG.git
cd Multimodal-Healthcare-RAG
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Create `.env` file and add your API keys**
```
COHERE_API_KEY=your-cohere-key-here
GROQ_API_KEY=your-groq-key-here
```

**5. Run the app**
```bash
python -m streamlit run app.py
```

---

## How it Works

```
User uploads file
      |
      v
Detect type (PDF or Image)
      |
      |-- PDF --> Extract text --> Split by sections
      |
      |-- Image --> Vision LLM --> Text description
      |
      v
Embed chunks with Cohere
      |
      v
Store in ChromaDB (per session)
      |
      v
User asks a question
      |
      v
Embed question --> Search ChromaDB --> Retrieve top chunks
      |
      v
Send chunks + question to Groq LLM
      |
      v
Plain language answer
```

---

## Sample Questions to Try

- "What is my haemoglobin level?"
- "Is my Vitamin D deficient?"
- "What are my abnormal values?"
- "Explain my lipid profile"
- "Is my blood sugar normal?"
- "What does my X-ray show?" (for images)

---

## Disclaimer

MediScan AI is built as a **personal project for educational purposes only**.

This tool is designed to help people better understand their medical reports 
in simple language — it is not a replacement for professional medical advice. 
The information provided may not always be fully accurate, as AI can make mistakes.

Please always consult a qualified doctor before making any health-related decisions.

---

## Author

**Sanvi Sharma**  
[GitHub](https://github.com/sanvisharma29) | [LinkedIn](https://linkedin.com/in/sanvisharma29)