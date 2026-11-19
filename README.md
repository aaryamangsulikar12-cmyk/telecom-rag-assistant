# 📡 Telecom Tariff & Policy Knowledge Assistant (Local RAG)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-00599C)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

A privacy-first **Retrieval-Augmented Generation (RAG)** assistant designed for the telecommunications domain. It ingests regulatory tariff orders, international roaming terms, and consumer handbooks to answer customer and compliance queries with verifiable **document and page-level citations**—running 100% locally via **Ollama** with zero external API dependencies.

---

## 🏗️ System Architecture

```text
[Telecom PDFs (TRAI / Airtel)]
        │
        ▼  PyPDFLoader + RecursiveCharacterTextSplitter (chunk_size=900, overlap=150)
[Text Chunks + Metadata]
        │
        ▼  Ollama Embeddings (nomic-embed-text)
[Local FAISS Vector Index] ◄─── Semantic Similarity Search (Top-K = 4) ─── [User Query]
        │
        ▼  Retrieved Context + Grounding Prompt
[Ollama Chat LLM (llama3.2, temp=0.2)]
        │
        ▼
[Streamlit Chat UI + Expandable Page Citations]

## ✨ Key Features
- **100% Local & Privacy-First:** Uses **Ollama** (`nomic-embed-text` for embeddings and `llama3.2` for generation) with zero external API keys or data leakage.
- **Source & Page Citations:** Every answer displays the exact source PDF filename, page number, and text snippet retrieved from FAISS.
- **Persistent Vector Index:** Automatically caches the FAISS vector store on disk for fast startup.
- **Live PDF Ingestion:** Upload additional telecom policy PDFs directly from the Streamlit sidebar to re-index on the fly.

## 🛠️ Tech Stack
- **LLM & Embeddings:** Ollama (`llama3.2`, `nomic-embed-text`)
- **RAG Orchestration:** LangChain (LCEL)
- **Vector Store:** FAISS (`faiss-cpu`)
- **Document Parsing:** PyPDF (`RecursiveCharacterTextSplitter`)
- **Frontend UI:** Streamlit

git clone [https://github.com/](https://github.com/)<your-username>/telecom-rag-assistant.git
cd telecom-rag-assistant

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
