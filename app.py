from pathlib import Path
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---
DATA_DIR = Path("data")
FAISS_DIR = Path("faiss_index")
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"

# Configure browser tab title and layout
st.set_page_config(page_title="Telecom Plan & Policy Assistant", page_icon="📡", layout="wide")
st.title("📡 Telecom Plan & Policy Knowledge Assistant")
st.caption("Ask questions about prepaid/postpaid tariffs, international roaming rules, MNP, and TRAI consumer guidelines.")


# --- Helper: Build or Load Vector Store ---
# @st.cache_resource ensures the FAISS index loads once and stays in memory across chat messages
@st.cache_resource
def get_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Load existing FAISS index if already created
    if FAISS_DIR.exists():
        return FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    # Otherwise, build it from PDFs inside data/
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        return None

    all_docs = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    FAISS_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))
    return vectorstore


# Helper to format retrieved chunks with filename & page number for the LLM prompt
def format_docs_with_sources(docs):
    formatted = []
    for i, doc in enumerate(docs, start=1):
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", 0) + 1
        formatted.append(f"[Source {i}: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


# --- Sidebar: Indexed Documents & Controls ---
with st.sidebar:
    st.header("📂 Knowledge Base")
    pdf_list = sorted(DATA_DIR.glob("*.pdf"))
    if pdf_list:
        st.write(f"**{len(pdf_list)} Telecom PDFs Indexed:**")
        for pdf in pdf_list:
            st.markdown(f"- `{pdf.name}`")
    else:
        st.warning("No PDFs found in `data/`.")

    st.divider()

    # Optional: Upload a new PDF right from the UI and add it to FAISS
    uploaded_file = st.file_uploader("Add another telecom PDF", type=["pdf"])
    if uploaded_file is not None:
        DATA_DIR.mkdir(exist_ok=True)
        save_path = DATA_DIR / uploaded_file.name
        if not save_path.exists():
            save_path.write_bytes(uploaded_file.getvalue())
            # Clear cached vectorstore and rebuild index with the new file
            if FAISS_DIR.exists():
                for f in FAISS_DIR.iterdir():
                    f.unlink()
                FAISS_DIR.rmdir()
            st.cache_resource.clear()
            st.success(f"Added `{uploaded_file.name}`! Re-indexing...")
            st.rerun()

    # Button to clear chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# --- Initialize RAG Chain ---
vectorstore = get_vectorstore()
if vectorstore is None:
    st.error("Please add at least one PDF to the `data/` folder or upload one in the sidebar.")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_template(
    """You are a helpful telecom policy and tariff assistant.
Answer the question using ONLY the provided context.
If the answer is not clearly in the context, say "I couldn't find that information in the provided telecom documents."
Mention the document name and page number for key claims when available.

Context:
{context}

Question:
{question}

Answer:"""
)

llm = ChatOllama(model=CHAT_MODEL, temperature=0.2)

rag_chain = (
    {
        "context": retriever | format_docs_with_sources,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# --- Chat History State ---
# Store conversation messages across Streamlit reruns
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me anything about the telecom tariff orders, international roaming packs, or TRAI consumer guidelines.",
            "sources": [],
        }
    ]

# Render previous messages in the chat window
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 View Retrieved Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['file']}** (Page {src['page']}): _{src['snippet']}..._")


# --- Chat Input & Response Generation ---
if user_query := st.chat_input("Ask a question about roaming, tariffs, MNP, or consumer rules..."):
    # 1. Display user message immediately
    st.session_state.messages.append({"role": "user", "content": user_query, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Retrieve matching PDF chunks and generate grounded answer
    with st.chat_message("assistant"):
        with st.spinner("Searching telecom documents..."):
            retrieved_docs = retriever.invoke(user_query)
            answer = rag_chain.invoke(user_query)

            # Collect clean source citations
            sources_info = []
            for doc in retrieved_docs:
                sources_info.append(
                    {
                        "file": Path(doc.metadata.get("source", "unknown")).name,
                        "page": doc.metadata.get("page", 0) + 1,
                        "snippet": doc.page_content[:180].replace("\n", " "),
                    }
                )

            st.markdown(answer)
            with st.expander("📄 View Retrieved Sources"):
                for src in sources_info:
                    st.markdown(f"- **{src['file']}** (Page {src['page']}): _{src['snippet']}..._")

    # 3. Save assistant response & sources to session history
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources_info}
    )